import json
import os.path as osp

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC

import gcforest.data_load as load


def feat_indx(database_name, top, threhold=0.0001):
    output_dir = osp.join("output", "result")
    path = osp.join(output_dir, database_name)
    file = open(path, 'r')
    dicts = json.load(file)

    for key, value in dicts.iteritems():
        if key == str(threhold):
            df = pd.DataFrame({'feature': value.keys(), 'importance': value.values()})
            df = df.sort_values(by=['importance'], ascending=False)

            feat_idx = df['feature'].tolist()

            feat_idx = [int(f) for f in feat_idx]
            return feat_idx[:top]


if __name__ == "__main__":
    cv = StratifiedKFold(n_splits=5, shuffle=False, random_state=0)
    # clf_rf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf_svm = SVC(kernel='linear', C=1,
                  gamma=0.001, random_state=0, probability=True)

    datasets = ['cirrhosis', 't2d', 'obesity']
    top_feat = [50]
    name = 'obesity'
    X, Y = load.t2d_data()
    for top in top_feat:
        feat_idx = feat_indx(name, top)
        X_hat = X.ix[:, feat_idx]

        clf_acc_before = cross_val_score(clf_svm, X, Y, cv=cv, scoring='accuracy')
        clf_acc_after = cross_val_score(clf_svm, X_hat, Y, cv=cv, scoring='accuracy')
        print(name, top, np.mean(clf_acc_before), np.mean(clf_acc_after))
