import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets
from sklearn.preprocessing import normalize
from joblib import Parallel, delayed
from numpy import unique
import math
import multiprocessing
from lib.gcforest.datasets.graph import Graph
from sklearn.model_selection import train_test_split


def obesity_data():
    f = pd.read_csv('../lib/gcforest/data/obesity/count_matrix.csv', sep=',', header=None)
    f = f.loc[(f != 0).any(axis=1)]
    f = (f - f.min()) / (f.max() - f.min())

    l = pd.read_csv('../lib/gcforest/data/obesity/labels.txt', sep='\t', header=None)
    l = l.T.iloc[0]
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(l)

    return f, integer_encoded


def cirrhosis_data():
    f = pd.read_csv('../lib/gcforest/data/cirrhosis/count_matrix.csv', sep=',', header=None)
    f = f.loc[(f != 0).any(axis=1)]
    f = (f - f.min()) / (f.max() - f.min())

    l = pd.read_csv('../lib/gcforest/data/cirrhosis/labels.txt', sep='\t', header=None)
    l = l.T.iloc[0]
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(l)

    return f, integer_encoded


def t2d_data():
    f = pd.read_csv('../lib/gcforest/data/t2d/count_matrix.csv', sep=',', header=None)
    f = f.loc[(f != 0).any(axis=1)]
    f = (f - f.min()) / (f.max() - f.min())

    l = pd.read_csv('../lib/gcforest/data/t2d/labels.txt', sep='\t', header=None)
    l = l.T.iloc[0]
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(l)

    return f, integer_encoded


def yeast_data():
    f = pd.read_fwf('../lib/gcforest/data/yeast.data', header=None)
    features = f.iloc[:, 1:9]
    labels = f.iloc[:, 9]

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    return features, integer_encoded


def breast_cancer_data():
    breast_cancer = datasets.load_breast_cancer()
    X = breast_cancer.data
    y = breast_cancer.target
    return pd.DataFrame(X), y


# convert abundance vector into tree matrix
def _generate_maps(x, g, f):
    g.populate_graph(f, x)
    return x, np.array(g.get_map())


def load_obesity():
    data_dir = "../lib/gcforest/data/obesity"

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
    # my_benchmark = []

    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(_generate_maps)(x, g, features) for x in my_data)
    my_maps.append(np.array(np.take(results, 1, 1).tolist()))
    # my_benchmark.append(np.array(np.take(results, 0, 1).tolist()))

    my_maps = np.array(my_maps)
    # my_benchmark = np.array(my_benchmark)
    map_rows = my_maps.shape[2]
    map_cols = my_maps.shape[3]

    X = my_maps[0].reshape(-1, map_rows, map_cols)

    x_tr, x_te, y_tr, y_te = train_test_split(X, my_lab, test_size=0.2, stratify=my_lab)

    y_train = np.squeeze(np.array(y_tr).reshape(1, -1), 0)
    y_test = np.squeeze(np.array(y_te).reshape(1, -1), 0)

    X_train = x_tr[:, np.newaxis, :, :]
    X_test = x_te[:, np.newaxis, :, :]

    return X_train, X_test, y_train, y_test
