# -*- coding:utf-8 -*-
import gcforest.data_load as load
import gcforest.data_load_phy as load2
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from gcforest.gcforest import GCForest
import numpy as np
import os, os.path as osp
from xgboost import XGBClassifier

output_dir = osp.join("output", "result")
if not osp.exists(output_dir):
    os.makedirs(output_dir)
file = osp.join(output_dir, "features_selection.txt")
with open(file, 'w') as wf:
    wf.write('======================\n')


def load_json(path):
    import json
    """
    支持以//开头的注释
    """
    lines = []
    with open(path) as f:
        for row in f.readlines():
            if row.strip().startswith("//"):
                continue
            lines.append(row)
    return json.loads("\n".join(lines))


def gcforest_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 4
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 10, "n_jobs": -1, "max_features": 1,
         "random_state": 0})
    # ca_config["estimators"].append(
    #     {"n_folds": 3, "type": "RandomForestClassifier", "n_estimators": 10, "n_jobs": -1, "max_features":1,"random_state":0})
    # ca_config["estimators"].append(
    #     {"n_folds": 3, "type": "RandomForestClassifier", "n_estimators": 10, "n_jobs": -1, "max_features": 1,"random_state":0})
    # ca_config["estimators"].append(
    #     {"n_folds": 3, "type": "RandomForestClassifier", "n_estimators": 10, "n_jobs": -1, "max_features": 1,"random_state":0})
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 10, "n_jobs": -1, "random_state": 0})
    # ca_config["estimators"].append(
    #     {"n_folds": 3, "type": "RandomForestClassifier", "n_estimators": 10, "n_jobs": -1,"random_state":0})
    # ca_config["estimators"].append(
    #     {"n_folds": 3, "type": "RandomForestClassifier", "n_estimators": 10, "n_jobs": -1,"random_state":0})
    # ca_config["estimators"].append(
    #     {"n_folds": 3, "type": "RandomForestClassifier", "n_estimators": 10, "n_jobs": -1,"random_state":0})
    # ca_config["estimators"].append(
    #         {"n_folds": 3, "type": "XGBClassifier", "n_estimators": 10,
    #           "silent": True, "nthread": -1, "learning_rate": 0.1} )
    # ca_config["estimators"].append({"n_folds": 3, "type": "ExtraTreesClassifier","max_depth": None, "n_jobs": -1})
    # ca_config["estimators"].append({"n_folds": 3, "type": "LogisticRegression"})
    config["cascade"] = ca_config
    return config


def write_output_results(content):
    with open(file, 'a') as wf:
        wf.write(content)


def write_final_important_features(clf):
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    num_selected_features = 30

    indices = indices[:num_selected_features]

    output_feat = []
    for f in range(0, num_selected_features):
        f = "%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]])
        output_feat.append(f)
    if len(output_feat) > 0:
        content = "\n".join(output_feat)
        file = osp.join(output_dir, "features_selection.txt")
        with open(file, 'a') as wf:
            wf.write(content)


if __name__ == "__main__":
    X, Y = load2.cirrhosis_data()

    clf_rf = RandomForestClassifier(n_estimators=100, random_state=0)

    AUCs = []
    for i in range(10):
        cv = StratifiedKFold(n_splits=10, shuffle=True)
        scores = cross_val_score(clf_rf, X, Y, cv=cv, scoring='accuracy')
        print("clf_rf", scores)
        print("Accuracy of Random Forest Classifier: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

        # # ==============================================
        f, ax = plt.subplots(1, 1)
        params = [(clf_rf, 'green', "Random Forest")]
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []
        for train, test in cv.split(X, Y):
            clf_rf.fit(X.iloc[train], Y[train])
            probas_ = clf_rf.predict_proba(X.iloc[test])
            fpr, tpr, thresholds = roc_curve(Y[test], probas_[:, 1])
            v = interp(mean_fpr, fpr, tpr)
            tprs.append(v)
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        print("mean_auc=" + str(mean_auc))
        AUCs.append(mean_auc)
        # std_auc = np.std(aucs)
        # ax.plot(mean_fpr, mean_tpr, color='green', label='Random Forest' '(auc = {:.3f})'.format(mean_auc), lw=2,
        #         alpha=.8)
    #
    # ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', alpha=.8)
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.legend(loc='lower right')
    # plt.show()
    print("Accuracy Deep Forest: %0.3f (+/- %0.3f)" % (np.array(AUCs).mean(), np.array(AUCs).std() * 2))
