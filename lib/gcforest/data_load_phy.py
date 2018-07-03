import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets
from sklearn.preprocessing import normalize


def obesity_data():
    f = pd.read_csv('../lib/gcforest/data/obesity/count_matrix.csv', sep=',', header=None)
    f = f.loc[(f != 0).any(axis=1)]
    # f = (f - f.min()) / (f.max() - f.min())

    l = pd.read_csv('../lib/gcforest/data/obesity/labels.txt', sep='\t', header=None)
    l = l.T.iloc[0]
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(l)

    return f, integer_encoded


def cirrhosis_data():
    f = pd.read_csv('../lib/gcforest/data/cirrhosis/count_matrix.csv', sep=',', header=None)
    f = f.loc[(f != 0).any(axis=1)]
    # f = (f - f.min()) / (f.max() - f.min())

    l = pd.read_csv('../lib/gcforest/data/cirrhosis/labels.txt', sep='\t', header=None)
    l = l.T.iloc[0]
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(l)

    return f, integer_encoded


def t2d_data():
    f = pd.read_csv('../lib/gcforest/data/t2d/count_matrix.csv', sep=',', header=None)
    f = f.loc[(f != 0).any(axis=1)]
    # f = (f - f.min()) / (f.max() - f.min())

    l = pd.read_csv('../lib/gcforest/data/t2d/labels.txt', sep='\t', header=None)
    l = l.T.iloc[0]
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(l)

    return f, integer_encoded

def yeast_data():
    f = pd.read_fwf('../lib/gcforest/data/yeast.data', header=None)
    features = f.iloc[:,1:9]
    labels = f.iloc[:,9]

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    return features, integer_encoded


def breast_cancer_data():
    breast_cancer = datasets.load_breast_cancer()
    X = breast_cancer.data
    y = breast_cancer.target
    return pd.DataFrame(X), y
