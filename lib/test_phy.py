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

import gcforest.data_load_phy as load2


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

    config1 = load_json("/home/qiang/repo/python/experiment-gcForest/cascade_clf/examples/demo_gc_stacking.json")
    # If the model you use cost too much memory for you.
    # You can use these methods to force gcforest not keeping model in memory
    # gc.set_keep_model_in_mem(False), default is TRUE.
    config2 = load_json("/home/qiang/repo/python/experiment-gcForest/cascade_clf/examples/demo_gc.json")
    acc_st = []
    acc_gc = []


    for i in range(10):
        X_train, X_test, y_train, y_test = load2.load_obesity()


        gc1 = GCForest(config1)
        gc1.fit_transform(X_train, y_train)
        y_pred = gc1.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        acc_st.append(acc)
        print("Test Accuracy of stacking GcForest = {:.2f} %".format(acc * 100))


        gc2 = GCForest(config2)
        gc2.fit_transform(X_train, y_train)


        y_pred = gc2.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        acc_gc.append(acc)
        print("Test Accuracy of GcForest = {:.2f} %".format(acc * 100))

    print(acc_st)
    print(acc_gc)




