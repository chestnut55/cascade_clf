import gcforest.data_load as load
import gcforest.data_load_phy as load2
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import KFold,StratifiedKFold
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
with open(file,'w') as wf:
    wf.write('======================\n')

def gcforest_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 4
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
    # ca_config["estimators"].append(
    #     {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 10, "n_jobs": -1, "max_features":1})
    # ca_config["estimators"].append(
    #     {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 10, "n_jobs": -1, "max_features":1})
    # ca_config["estimators"].append(
    #     {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 10, "n_jobs": -1, "max_features": 1})
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 10, "n_jobs": -1, "max_features": 1})
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 10, "n_jobs": -1})
    # ca_config["estimators"].append(
    #     {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 10, "n_jobs": -1})
    # ca_config["estimators"].append(
    #     {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 10, "n_jobs": -1})
    # ca_config["estimators"].append(
    #     {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 10, "n_jobs": -1})
    # ca_config["estimators"].append(
    #         {"n_folds": 5, "type": "XGBClassifier", "n_estimators": 10,
    #           "silent": True, "nthread": -1, "learning_rate": 0.1} )
    # ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier","max_depth": None, "n_jobs": -1})
    # ca_config["estimators"].append({"n_folds": 5, "type": "LogisticRegression"})
    config["cascade"] = ca_config
    return config

if __name__ == "__main__":
    X, Y = load2.cirrhosis_data()


    cv = StratifiedKFold(n_splits=5,shuffle=False)

    clf_rf = RandomForestClassifier(n_estimators=100)
    scores = cross_val_score(clf_rf, X, Y, cv=cv, scoring='accuracy')
    print("clf_rf", scores)
    print("Accuracy of Random Forest Classifier: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

    clf_svm = SVC(kernel='linear', C=1,
                  gamma=0.001, random_state=0, probability=True)
    scores = cross_val_score(clf_svm, X, Y, cv=cv, scoring='accuracy')
    print("clf_svm", scores)
    print("Accuracy SVM Classifier: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

    xgb_crf = XGBClassifier(n_estimators=100)
    scores = cross_val_score(xgb_crf, X, Y, cv=cv, scoring='accuracy')
    print("xgb_crf", scores)
    print("Accuracy of extreme gradient boosting Classifier: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

    config = gcforest_config()
    clf_gc = GCForest(config)
    gc_pred_acc = []
    gc_pred_acc_clf = []

    # # =============================================
    for train, test in cv.split(X, Y):
        probas_ = None

        x_train = X.iloc[train]
        y_train = Y[train]

        x_test = X.iloc[test]
        y_test = Y[test]

        x_train = x_train.values.reshape(-1, 1, len(x_train.columns))
        x_test = x_test.values.reshape(-1, 1, len(x_test.columns))

        X_train = x_train[:, np.newaxis, :, :]
        X_test = x_test[:, np.newaxis, :, :]

        X_train_enc = clf_gc.fit_transform(X_train, y_train)

        ###############################
        y_pred = clf_gc.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        gc_pred_acc.append(acc)
        ###########################################################
        X_test_enc = clf_gc.transform(X_test)
        X_train_enc = X_train_enc.reshape((X_train_enc.shape[0], -1))
        X_test_enc = X_test_enc.reshape((X_test_enc.shape[0], -1))
        X_train_origin = X_train.reshape((X_train.shape[0], -1))
        X_test_origin = X_test.reshape((X_test.shape[0], -1))
        X_train_enc = np.hstack((X_train_enc, X_train_origin))
        X_test_enc = np.hstack((X_test_enc, X_test_origin))
        clf = XGBClassifier(n_estimators=100, n_jobs=-1)
        clf.fit(X_train_enc, y_train)

        y_pred = clf.predict(X_test_enc)
        acc = accuracy_score(y_test, y_pred)
        gc_pred_acc_clf.append(acc)

    print("gc_pred_acc", gc_pred_acc)
    print("Accuracy Deep Forest: %0.3f (+/- %0.3f)" % (np.array(gc_pred_acc).mean(), np.array(gc_pred_acc).std() * 2))

    print("gc_pred_acc_clf", gc_pred_acc_clf)
    print("Accuracy Deep Forest: %0.3f (+/- %0.3f)" % (np.array(gc_pred_acc_clf).mean(), np.array(gc_pred_acc_clf).std() * 2))

