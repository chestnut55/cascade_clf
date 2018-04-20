from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
from mlxtend.classifier import StackingClassifier

import gcforest.data_load_phy as load2

if __name__ == "__main__":
    X, Y = load2.obesity_data()

    rf = RandomForestClassifier(n_estimators=100, random_state=1)

    et = ExtraTreesClassifier(n_estimators=100, random_state=1)

    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=1)

    lr = LogisticRegression()
    eclf = StackingClassifier(classifiers=[rf, et, xgb], meta_classifier=lr)

    for clf, label in zip([rf, et, xgb, eclf], ['Random Forest', 'ExtraTreesClassifier', 'XGBClassifier', 'Ensemble']):
        scores = cross_val_score(clf, X, Y, cv=5, scoring='accuracy')
        print("Accuracy: %0.2f (+/-) %0.2f [%s]" % (scores.mean(), scores.std(), label))
