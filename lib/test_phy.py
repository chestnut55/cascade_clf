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
    ca_config["estimators"].append(
        {"n_folds": 3, "type": "ExtraTreesClassifier", "n_estimators": 50, "n_jobs": -1, "max_features": 1})
    ca_config["estimators"].append(
        {"n_folds": 3, "type": "ExtraTreesClassifier", "n_estimators": 50, "n_jobs": -1, "max_features": 1})
    # ca_config["estimators"].append(
    #     {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 10, "n_jobs": -1})
    # ca_config["estimators"].append(
    #     {"n_folds": 3, "type": "RandomForestClassifier", "n_estimators": 10, "n_jobs": -1})
    ca_config["estimators"].append(
        {"n_folds": 3, "type": "RandomForestClassifier", "n_estimators": 50, "n_jobs": -1})
    ca_config["estimators"].append(
        {"n_folds": 3, "type": "RandomForestClassifier", "n_estimators": 50, "n_jobs": -1})
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

    args = parse_args()
    config = load_json(args.model)
    # config = gcforest_config()
    clf_gc = GCForest(config)


