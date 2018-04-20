import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.ensemble import BaseEnsemble

import gcforest.data_load_phy as load2


class CascadeForest():
    def __init__(self, estimators_config, folds=3):
        self.estimators_config = estimators_config
        self.folds = folds

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
            X = np.hstack([X] + predictions)
            y_prediction = self.classes.take(
                np.array(predictions).mean(axis=0).argmax(axis=1)
            )
            score = accuracy_score(y, y_prediction)
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

    def print_important_features(self, clf):
        if isinstance(clf, BaseEnsemble):
            importances = clf.feature_importances_
            indices = np.argsort(importances)[::-1]
            num_selected_features = len(indices)

            indices = indices[:num_selected_features]

            for f in range(0, num_selected_features):
                f = "%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]])
                print(f)

def print_important_features(clf):
    if isinstance(clf, BaseEnsemble):
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        num_selected_features = len(indices)

        indices = indices[:num_selected_features]

        for f in range(0, num_selected_features):
            f = "%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]])
            print(f)
if __name__ == "__main__":

    estimators_config = {
        'cascade': [{
            'estimator_class': RandomForestClassifier,
            'estimator_params': {
                'n_estimators': 180,
                'n_jobs': -1,
                'random_state': 42
            }
        }, {
            'estimator_class': RandomForestClassifier,
            'estimator_params': {
                'n_estimators': 180,
                'max_features': 1,
                'n_jobs': -1,
                'random_state': 42
            }
        }]
    }

    X, Y = load2.yeast_data()
    cv = StratifiedKFold(n_splits=5, shuffle=False, random_state=42)

    for i in range(2):
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []
        acc = []

        _name = None
        color = None
        for train, test in cv.split(X, Y):
            X_tr = X.iloc[train]
            y_tr = Y[train]

            X_te = X.iloc[test]
            y_te = Y[test]

            y_pred = None
            if i == 0:
                c_forest = CascadeForest(estimators_config['cascade'])
                c_forest.fit(X_tr, y_tr)

                y_pred = c_forest.predict(X_te)

                _name = 'cascade forest'
            elif i == 1:
                rf = RandomForestClassifier(n_estimators=180, random_state=42)
                rf.fit(X_tr, y_tr)
                print("===============================")
                # print_important_features(rf)
                y_pred = rf.predict(X_te)
                _name = 'random forest'

            acc.append(accuracy_score(y_te, y_pred))
        print(_name + " accuracy=" + str(np.mean(acc)))
