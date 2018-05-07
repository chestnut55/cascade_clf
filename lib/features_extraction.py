# -*- coding:utf-8 -*-
import json
import os
import os.path as osp
import pandas as pd

from sklearn.model_selection import StratifiedKFold

import gcforest.data_load as load
from gcforest.gcforest import GCForest
from gcforest.utils.log_utils import get_logger

LOGGER = get_logger('cascade_clf.lib.plot_roc_all')


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


def save_features(data_name, features):
    output_dir = osp.join("output", "result")
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    file = osp.join(output_dir, data_name)
    with open(file, 'w') as wf:
        wf.write(json.dumps(features))


if __name__ == "__main__":
    cv = StratifiedKFold(n_splits=5, shuffle=False, random_state=0)

    config = load_json("/home/qiang/repo/python/cascade_clf/examples/demo_ca.json")
    gc = GCForest(config)

    datasets = ['cirrhosis','t2d', 'obesity']
    importance_thresholds = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]

    for dataset_idx, name in enumerate(datasets):
        thre_features = {}
        for imp_idx, thre in enumerate(importance_thresholds):
            X = None
            Y = None
            if name == 'cirrhosis':
                X, Y = load.cirrhosis_data()
            elif name == 't2d':
                X, Y = load.t2d_data()
            elif name == 'obesity':
                X, Y = load.obesity_data()
            else:
                raise Exception('the dataset is not defined!!!')

            ca_features = pd.Series()
            for train, test in cv.split(X, Y):
                x_train = X.iloc[train]
                y_train = Y[train]

                x_test = X.iloc[test]
                y_test = Y[test]

                X_train = x_train.values.reshape(-1, 1, len(x_train.columns))
                X_test = x_test.values.reshape(-1, 1, len(x_test.columns))

                X_train_enc, _features = gc.fit_transform(X_train, y_train,threshold=thre)

                probas_ = gc.predict_proba(X_test)
                ca_features = ca_features.add(_features, fill_value=0)

            if len(ca_features) > 0:
                thre_features[str(thre)] = dict(ca_features)

        if len(thre_features) > 0:
            save_features(name, thre_features)
