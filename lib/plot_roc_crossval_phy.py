# -*- coding:utf-8 -*-
import argparse
import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy import unique
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_curve,accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from gcforest.gcforest import GCForest
from lib.gcforest.datasets.graph import Graph


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", type=str, default=None, help="gcforest Net Model File")
    args = parser.parse_args()
    return args

def gcforest_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
    # ca_config["estimators"].append(
    #     {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 00, "n_jobs": -1})
    # ca_config["estimators"].append(
    #     {"n_folds": 3, "type": "RandomForestClassifier", "n_estimators": 10, "n_jobs": -1, "max_features":1})
    # ca_config["estimators"].append(
    #     {"n_folds": 3, "type": "RandomForestClassifier", "n_estimators": 10, "n_jobs": -1, "max_features": 1})
    # ca_config["estimators"].append(
    #     {"n_folds": 3, "type": "RandomForestClassifier", "n_estimators": 10, "n_jobs": -1, "max_features": 1})
    # ca_config["estimators"].append(
    #     {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 10, "n_jobs": -1})
    # ca_config["estimators"].append(
    #     {"n_folds": 3, "type": "RandomForestClassifier", "n_estimators": 10, "n_jobs": -1})
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 100, "n_jobs": -1})
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 100, "n_jobs": -1})
    # ca_config["estimators"].append(
    #         {"n_folds": 5, "type": "XGBClassifier", "n_estimators": 10,
    #           "silent": True, "nthread": -1, "learning_rate": 0.1} )
    # ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier","n_estimators": 100, "max_depth": None, "n_jobs": -1})
    # ca_config["estimators"].append({"n_folds": 5, "type": "LogisticRegression"})
    config["cascade"] = ca_config
    return config

# convert abundance vector into tree matrix
def generate_maps(x, g, f):
    g.populate_graph(f, x)
    return x, np.array(g.get_map())


if __name__ == "__main__":
    data_name = "cirrhosis"
    data_dir = "../lib/gcforest/data/" + data_name

    my_x = []
    my_y = []

    my_x = np.loadtxt(data_dir + '/count_matrix.csv', dtype=np.float32, delimiter=',')

    my_x = (my_x - my_x.min()) / (my_x.max() - my_x.min())

    my_y = np.genfromtxt(data_dir + '/labels.txt', dtype=np.str_, delimiter=',')
    features = np.genfromtxt(data_dir + '/otu.csv', dtype=np.str_, delimiter=',')

    num_samples = my_x.shape[0]
    num_features = len(my_x[0])
    classes = list(unique(my_y))
    num_classes = len(classes)

    my_ref = pd.factorize(my_y)[1]
    f = open(data_dir + "/label_reference.txt", 'w')
    f.write(str(my_ref))
    f.close()

    g = Graph()
    g.build_graph(data_dir + "/newick.txt")

    my_data = pd.DataFrame(my_x)
    my_data = np.array(my_data)
    my_lab = pd.factorize(my_y)[0]
    my_maps = []
    my_benchmark = []

    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(generate_maps)(x, g, features) for x in my_data)
    my_maps.append(np.array(np.take(results, 1, 1).tolist()))
    my_benchmark.append(np.array(np.take(results, 0, 1).tolist()))

    my_maps = np.array(my_maps)
    my_benchmark = np.array(my_benchmark)
    map_rows = my_maps.shape[2]
    map_cols = my_maps.shape[3]

    cv = StratifiedKFold(n_splits=5, shuffle=False,random_state=0)

    clf_rf = RandomForestClassifier(n_estimators=100, random_state=0)

    clf_svm = SVC(kernel='linear', C=1, gamma=0.001, random_state=0, probability=True)

    args = parse_args()
    config = load_json(args.model)
    # config = gcforest_config()
    clf_gc = GCForest(config)

    f, ax = plt.subplots(1, 1)
    params = [(clf_rf, 'green', "Random Forest"),
              (clf_svm, 'black', "SVM"),
              (clf_gc, 'red', "Deep Forest")]

    accs = []
    for x in params:
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []
        for train_index, test_index in cv.split(my_maps[0], my_lab):
            x_train = []
            x_test = []
            y_train = []
            y_test = []

            benchmark_train = []
            benchmark_test = []

            x_train.append(my_maps[0][train_index])
            x_test.append(my_maps[0][test_index])
            y_train.append(my_lab[train_index])
            y_test.append(my_lab[test_index])

            benchmark_train.append(my_benchmark[0][train_index])
            benchmark_test.append(my_benchmark[0][test_index])

            x_train = np.array(x_train).reshape(-1, map_rows, map_cols)
            x_test = np.array(x_test).reshape(-1, map_rows, map_cols)
            y_train = np.squeeze(np.array(y_train).reshape(1, -1), 0)
            y_test = np.squeeze(np.array(y_test).reshape(1, -1), 0)

            benchmark_train = np.array(benchmark_train).reshape(-1, num_features)
            benchmark_test = np.array(benchmark_test).reshape(-1, num_features)

            if isinstance(x[0], GCForest):
                gc = x[0]

                # x_train = x_train.values.reshape(-1, 1, len(x_train.columns)).astype('float32')
                # x_test = x_test.values.reshape(-1, 1, len(x_test.columns)).astype('float32')

                X_train = x_train[:, np.newaxis, :, :]
                X_test = x_test[:, np.newaxis, :, :]

                X_train_enc = gc.fit_transform(X_train, y_train)
                ####################################
                # gc.fit_transform(X_train, y_train)
                # probas_ = gc.predict_proba(X_test)

                ####################################
                X_test_enc = gc.transform(X_test)
                X_train_enc = X_train_enc.reshape((X_train_enc.shape[0], -1))
                X_test_enc = X_test_enc.reshape((X_test_enc.shape[0], -1))
                X_train_origin = X_train.reshape((X_train.shape[0], -1))
                X_test_origin = X_test.reshape((X_test.shape[0], -1))
                X_train_enc = np.hstack((X_train_enc, X_train_origin))
                X_test_enc = np.hstack((X_test_enc, X_test_origin))
                clf = RandomForestClassifier(n_estimators=1000, max_depth=10, n_jobs=-1)
                clf.fit(X_train_enc, y_train)

                ### output the important features
                y_pred = clf.predict(X_test_enc)
                probas_ = clf.predict_proba(X_test_enc)
                acc = accuracy_score(y_test, gc.predict(X_test))
                accs.append(acc)
            else:
                x[0].fit(benchmark_train, y_train)
                probas_ = x[0].predict_proba(benchmark_test)
            fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color=x[1], label='{}' '(auc = {:.3f})'.format(x[2], mean_auc), lw=2,
                alpha=.8)

    print("Accuracy Deep Forest: %0.3f (+/- %0.3f)" % (np.array(accs).mean(), np.array(accs).std() * 2))
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', alpha=.8)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc='lower right')
    plt.show()
