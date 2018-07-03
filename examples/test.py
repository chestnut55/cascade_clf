"""
MNIST datasets demo for gcforest
Usage:
    define the model within scripts:
        python examples/demo_mnist.py
    get config from json file:
        python examples/demo_mnist.py --model examples/demo_mnist-gc.json
        python examples/demo_mnist.py --model examples/demo_mnist-ca.json
"""
import argparse
import numpy as np
import sys
import os
from keras.datasets import mnist
import pickle
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold,cross_val_score
sys.path.insert(0, "/home/qiang/repo/python/experiment-gcForest/cascade_clf/lib")

from gcforest.gcforest import GCForest
from gcforest.utils.config_utils import load_json

from gcforest.datasets import uci_yeast


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", type=str, default=None, help="gcfoest Net Model File")
    args = parser.parse_args()
    return args


def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 2
    ca_config["n_classes"] = 10
    ca_config["estimators"] = []
    # ca_config["estimators"].append(
    #         {"n_folds": 5, "type": "XGBClassifier", "n_estimators": 10, "max_depth": 5,
    #          "objective": "multi:softprob", "silent": True, "nthread": -1, "learning_rate": 0.1} )
    # ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    # ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    # ca_config["estimators"].append({"n_folds": 5, "type": "LogisticRegression"})
    ca_config["estimators"].append(
        {"n_folds": 3, "type": "RandomForestClassifier", "n_estimators": 500, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append(
        {"n_folds": 3, "type": "RandomForestClassifier", "n_estimators": 500, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append(
        {"n_folds": 3, "type": "RandomForestClassifier", "n_estimators": 500, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append(
        {"n_folds": 3, "type": "RandomForestClassifier", "n_estimators": 500, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append(
        {"n_folds": 3, "type": "ExtraTreesClassifier", "n_estimators": 500, "max_depth": None, "n_jobs": -1,"max_features":1})
    ca_config["estimators"].append(
        {"n_folds": 3, "type": "ExtraTreesClassifier", "n_estimators": 500, "max_depth": None, "n_jobs": -1,"max_features":1})
    ca_config["estimators"].append(
        {"n_folds": 3, "type": "ExtraTreesClassifier", "n_estimators": 500, "max_depth": None, "n_jobs": -1,"max_features":1})
    ca_config["estimators"].append(
        {"n_folds": 3, "type": "ExtraTreesClassifier", "n_estimators": 500, "max_depth": None, "n_jobs": -1,"max_features":1})
    config["cascade"] = ca_config
    return config


if __name__ == "__main__":
    # args = parse_args()
    # if args.model is None:
    #     config = get_toy_config()
    # else:
    #     config = load_json(args.model)

    config1 = load_json("/home/qiang/repo/python/experiment-gcForest/cascade_clf/examples/demo_ca.json")
    # If the model you use cost too much memory for you.
    # You can use these methods to force gcforest not keeping model in memory
    # gc.set_keep_model_in_mem(False), default is TRUE.
    config2 = get_toy_config()
    acc_st = []
    acc_gc = []
    acc_rf = []
    for i in range(10):
        (X_train, y_train), (X_test, y_test) = uci_yeast.load_data()


        gc1 = GCForest(config1)
        gc1.fit_transform(X_train, y_train)
        y_pred = gc1.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        acc_st.append(acc)
        print("Test Accuracy of stacking GcForest = {:.2f} %".format(acc * 100))

        # X_train, y_train = X_train[:2000], y_train[:2000]
        # X_train = X_train[:, np.newaxis, :]
        # X_test = X_test[:, np.newaxis, :]

        gc2 = GCForest(config2)
        gc2.fit_transform(X_train, y_train)
        # X_enc is the concatenated predict_proba result of each estimators of the last layer of the GCForest model
        # X_enc.shape =
        #   (n_datas, n_estimators * n_classes): If cascade is provided
        #   (n_datas, n_estimators * n_classes, dimX, dimY): If only finegrained part is provided
        # You can also pass X_test, y_test to fit_transform method, then the accracy on test data will be logged when training.
        # X_train_enc, X_test_enc = gc.fit_transform(X_train, y_train, X_test=X_test, y_test=y_test)
        # WARNING: if you set gc.set_keep_model_in_mem(True), you would have to use
        # gc.fit_transform(X_train, y_train, X_test=X_test, y_test=y_test) to evaluate your model.

        y_pred = gc2.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        acc_gc.append(acc)
        print("Test Accuracy of GcForest = {:.2f} %".format(acc * 100))

        rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        acc_rf.append(acc)
        print("Test Accuracy of RandomForest = {:.2f} %".format(acc * 100))
    print(acc_st)
    print(acc_gc)
    print(acc_rf)



