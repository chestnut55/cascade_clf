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

if __name__ == "__main__":
    X, Y = load2.cirrhosis_data()


    config = gcforest_config()
    gc = GCForest(config)

    AUCs = []
    for i in range(10):
        cv = StratifiedKFold(n_splits=10, shuffle=True)
        # # ==============================================
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []
        for train, test in cv.split(X, Y):
            x_train = X.iloc[train]
            y_train = Y[train]

            x_test = X.iloc[test]
            y_test = Y[test]

            x_train = x_train.values.reshape(-1, 1, len(x_train.columns))
            x_test = x_test.values.reshape(-1, 1, len(x_test.columns))

            X_train = x_train[:, np.newaxis, :, :]
            X_test = x_test[:, np.newaxis, :, :]

            X_train_enc = gc.fit_transform(X_train, y_train)

            ###############################
            # y_pred = gc.predict(X_test)
            # acc = accuracy_score(y_test, y_pred)
            # gc_pred_acc.append(acc)
            # print("Test Accuracy of GcForest = {:.2f} %".format(acc * 100))
            # probas_ = gc.predict_proba(X_test)

            ###########################################################
            # You can try passing X_enc to another classfier on top of gcForest.e.g. xgboost/RF.
            X_test_enc = gc.transform(X_test)
            X_train_enc = X_train_enc.reshape((X_train_enc.shape[0], -1))
            X_test_enc = X_test_enc.reshape((X_test_enc.shape[0], -1))
            X_train_origin = X_train.reshape((X_train.shape[0], -1))
            X_test_origin = X_test.reshape((X_test.shape[0], -1))
            X_train_enc = np.hstack((X_train_enc, X_train_origin))
            X_test_enc = np.hstack((X_test_enc, X_test_origin))
            clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=0)
            clf.fit(X_train_enc, y_train)
            #
            # ### output the important features
            # # write_final_important_features(clf)

            y_pred = clf.predict(X_test_enc)
            acc = accuracy_score(y_test, y_pred)
            probas_ = clf.predict_proba(X_test_enc)
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

    print("Accuracy Deep Forest: %0.3f (+/- %0.3f)" % (np.array(AUCs).mean(), np.array(AUCs).std() * 2))
