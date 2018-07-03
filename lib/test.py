# -*- coding:utf-8 -*-
from sklearn.model_selection import StratifiedKFold,train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
import gcforest.data_load as load
import gcforest.data_load_phy as load2
from gcforest.gcforest import GCForest
from sklearn.metrics import accuracy_score
import numpy as np

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

X, Y = load.obesity_data()

x_tr,x_te,y_tr,y_te = train_test_split(X,Y,random_state=42,stratify=Y)

clf_rf = RandomForestClassifier(n_estimators=200, random_state=0)
clf_rf.fit(x_tr,y_tr)
y_pred = clf_rf.predict(x_te)
print(accuracy_score(y_te,y_pred))


config = load_json("/home/qiang/repo/python/cascade_clf/examples/demo_ca.json")
clf_gc = GCForest(config)

clf_gc.fit_transform(x_tr.values, y_tr)
y_pred = clf_gc.predict(x_te.values)
print(accuracy_score(y_te, y_pred))
