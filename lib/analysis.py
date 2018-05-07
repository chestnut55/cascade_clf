# -*- coding:utf-8 -*-
import json
import os
import os.path as osp
import pandas as pd
import gcforest.data_load as load
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from gcforest.gcforest import GCForest
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt


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


def feat_indx(database_name, threhold):
    output_dir = osp.join("output", "result")
    path = osp.join(output_dir, database_name)
    file = open(path, 'r')
    dicts = json.load(file)

    for key, value in dicts.iteritems():
        if key == str(threhold):
            df = pd.DataFrame({'feature': value.keys(), 'importance': value.values()})
            df = df.sort_values(['importance'])

            feat_idx = df['feature'].tolist()

            feat_idx = [int(f) for f in feat_idx]

            # if database_name == 'obesity':
            #     return feat_idx[:250]
            # elif database_name == 't2d':
            #     return feat_idx[:50]
            return feat_idx



def accuracy(clf, X, Y, cv):
    scores = cross_val_score(clf, X, Y, cv=cv, scoring='accuracy')
    return scores, scores.mean(), scores.std() * 2


def gc_acc(clf_gc, X, Y, cv):
    gc_pred_acc = []

    for train, test in cv.split(X, Y):
        x_train = X.iloc[train]
        y_train = Y[train]

        x_test = X.iloc[test]
        y_test = Y[test]

        X_train = x_train.values.reshape(-1, 1, len(x_train.columns))
        X_test = x_test.values.reshape(-1, 1, len(x_test.columns))
        clf_gc.fit_transform(X_train, y_train)

        y_pred = clf_gc.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        gc_pred_acc.append(acc)
        break
    return np.mean(gc_pred_acc)



def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height,
                "{:.3f}".format(height),
                ha='center', va='bottom')
if __name__ == '__main__':

    cv = StratifiedKFold(n_splits=5, shuffle=False, random_state=0)
    config = load_json("/home/qiang/repo/python/cascade_clf/examples/demo_ca.json")
    clf_gc = GCForest(config)

    datasets = ['cirrhosis', 't2d','obesity']
    classifiers = [(clf_gc, 'Deep Forest')]

    fig, ax = plt.subplots()
    avg_acc_before = []
    avg_acc_after = []
    for name in datasets:
        if name == 'cirrhosis':
            X, Y = load.cirrhosis_data()
            feat_idx = feat_indx(name, 0.001)
            X_hat = X.ix[:, feat_idx]
        elif name == 't2d':
            X, Y = load.t2d_data()
            feat_idx = feat_indx(name, 0.01)
            X_hat = X.ix[:, feat_idx]
        elif name == 'obesity':
            X, Y = load.obesity_data()
            feat_idx = feat_indx(name, 0.01)
            X_hat = X.ix[:, feat_idx]

        mean = gc_acc(clf_gc, X, Y, cv)
        avg_acc_before.append(mean)
        mean = gc_acc(clf_gc, X_hat, Y, cv)
        avg_acc_after.append(mean)

    n_groups = 3
    index = np.arange(n_groups)
    bar_width = 0.3
    opacity = 0.8

    rect1 = ax.bar(index, avg_acc_before, bar_width,
            alpha=opacity,
            color='red',
            label='Before')

    rect2 = ax.bar(index + bar_width, avg_acc_after, bar_width,
            alpha=opacity,
            color='green',
            label='After')

    plt.xticks(index + bar_width / 2, ('cirrhosis', 't2d','obesity'))
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 10})
    plt.ylabel('Accuracy')
    plt.title('Accuracy Before/After Feature Selection')
    plt.ylim([0, 1.0])

    autolabel(rect1)
    autolabel(rect2)
    plt.tight_layout(pad=8)
    plt.show()
