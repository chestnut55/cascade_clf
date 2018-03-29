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
import pickle
from keras.datasets import mnist
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from lib.gcforest.gcforest import GCForest
from lib.gcforest.utils.config_utils import load_json
import lib.gcforest.data_load as load
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
    # ca_config["estimators"].append(
    #         {"n_folds": 5, "type": "XGBClassifier", "n_estimators": 10, "max_depth": 5,
    #          "objective": "multi:softprob", "silent": True, "nthread": -1, "learning_rate": 0.1} )
    ca_config["estimators"].append({"n_folds": 1, "type": "RandomForestClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    # ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 100, "max_depth": None, "n_jobs": -1})
    # ca_config["estimators"].append({"n_folds": 5, "type": "LogisticRegression"})
    config["cascade"] = ca_config
    return config


if __name__ == "__main__":
    args = parse_args()
    if args.model is None:
        config = get_toy_config()
    else:
        config = load_json(args.model)

    gc = GCForest(config)
    # If the model you use cost too much memory for you.
    # You can use these methods to force gcforest not keeping model in memory
    # gc.set_keep_model_in_mem(False), default is TRUE.

    xs, ys = load.hmp_hmpii_data()
    X_train, X_test, y_train, y_test = train_test_split(xs, ys, test_size=0.1, random_state=42)

    clf = RandomForestClassifier(n_estimators=1000, max_depth=None, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Test Accuracy of RandomForestClassifier = {:.2f} %".format(acc * 100))

    X_train = X_train.values.reshape(-1, 1, len(X_train.columns)).astype('float32')
    X_test = X_test.values.reshape(-1, 1, len(X_test.columns)).astype('float32')

    # X_train, y_train = X_train[:2000], y_train[:2000]
    X_train = X_train[:, np.newaxis, :, :]
    X_test = X_test[:, np.newaxis, :, :]
    X_train_enc = gc.fit_transform(X_train, y_train)
    # X_enc is the concatenated predict_proba result of each estimators of the last layer of the GCForest model
    # X_enc.shape =
    #   (n_datas, n_estimators * n_classes): If cascade is provided
    #   (n_datas, n_estimators * n_classes, dimX, dimY): If only finegrained part is provided
    # You can also pass X_test, y_test to fit_transform method, then the accracy on test data will be logged when training.
    # X_train_enc, X_test_enc = gc.fit_transform(X_train, y_train, X_test=X_test, y_test=y_test)
    # WARNING: if you set gc.set_keep_model_in_mem(True), you would have to use
    # gc.fit_transform(X_train, y_train, X_test=X_test, y_test=y_test) to evaluate your model.
    
    y_pred = gc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Test Accuracy of GcForest = {:.2f} %".format(acc * 100))

    # You can try passing X_enc to another classfier on top of gcForest.e.g. xgboost/RF.
    X_test_enc = gc.transform(X_test)
    X_train_enc = X_train_enc.reshape((X_train_enc.shape[0], -1))
    X_test_enc = X_test_enc.reshape((X_test_enc.shape[0], -1))
    X_train_origin = X_train.reshape((X_train.shape[0], -1))
    X_test_origin = X_test.reshape((X_test.shape[0], -1))
    print("X_train_origin.shape={}, X_test_origin.shape={}".format(X_train_origin.shape, X_test_origin.shape))
    X_train_enc = np.hstack((X_train_origin, X_train_enc))
    X_test_enc = np.hstack((X_test_origin, X_test_enc))
    print("X_train_enc.shape={}, X_test_enc.shape={}".format(X_train_enc.shape, X_test_enc.shape))
    clf = RandomForestClassifier(n_estimators=1000, max_depth=None, n_jobs=-1)
    # clf = svm.SVC(kernel='linear', C=1, gamma=0.001, random_state=0, probability=True)
    # srhl_rbf = RBFRandomLayer(n_hidden=50, rbf_width=0.1, random_state=0)
    # clf = GenELMClassifier(hidden_layer=srhl_rbf)
    clf.fit(X_train_enc, y_train)
    y_pred = clf.predict(X_test_enc)
    acc = accuracy_score(y_test, y_pred)
    print("Test Accuracy of Other classifier using gcforest's X_encode = {:.2f} %".format(acc * 100))

    ############# plot feature selection###################################
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    print("indices=", len(indices))

    # Print the feature ranking
    print("Feature ranking:")
    # features = X_train_enc.shape[1]
    num_selected_features = 30
    print("features=", num_selected_features)

    indices = indices[:num_selected_features]
    for f in range(0, num_selected_features):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(num_selected_features), importances[indices], color="g",align="center")
    plt.xticks(range(num_selected_features), indices)
    plt.xlim([-1, num_selected_features])
    plt.show()
    #######################################################################


    # # dump
    # with open("test.pkl", "wb") as f:
    #     pickle.dump(gc, f, pickle.HIGHEST_PROTOCOL)
    # # load
    # with open("test.pkl", "rb") as f:
    #     gc = pickle.load(f)
    # y_pred = gc.predict(X_test)
    # acc = accuracy_score(y_test, y_pred)
    # print("Test Accuracy of GcForest (save and load) = {:.2f} %".format(acc * 100))

