from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
import gcforest.data_load as load
import numpy as np
import pandas as pd


def _features(clf, theshold):
    importances = clf.feature_importances_
    importances = [imp for imp in importances if imp > theshold]
    features_idx = np.argsort(importances)[::-1]


    return features_idx,importances

X, Y = load.obesity_data()
cv = StratifiedKFold(n_splits=5, shuffle=False, random_state=0)

clf_rf = RandomForestClassifier(n_estimators=50, random_state=0)
fea=[]


for train, test in cv.split(X, Y):
    clf_rf.fit(X.iloc[train], Y[train])
    feat, imp = _features(clf_rf, 0.005)
    # fea = list(set(fea).union(feat))

    aa = pd.DataFrame({'feat':feat,'imp':imp})
    aa = aa.sort_values('imp')
    print 'aaa'


