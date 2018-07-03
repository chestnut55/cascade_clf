import json
import os.path as osp

import matplotlib.pyplot as plt
import pandas as  pd

import gcforest.data_load as load


def feat_indx(database_name, threhold):
    output_dir = osp.join("output", "result")
    path = osp.join(output_dir, database_name)
    file = open(path, 'r')
    dicts = json.load(file)

    for key, value in dicts.iteritems():
        if key == str(threhold):
            df = pd.DataFrame({'feature': value.keys(), 'importance': value.values()})
            df = df.sort_values(by=['importance'], ascending=False)

            feat_idx = df['feature'].tolist()

            feat_idx = [int(f) for f in feat_idx]

            if database_name == 'obesity':
                return feat_idx[:50]
            elif database_name == 't2d':
                return feat_idx[:180]
            elif database_name == 'cirrhosis':
                return feat_idx[:80]
            return feat_idx





X, Y = load.cirrhosis_data()
feat_idx = feat_indx('cirrhosis', 0.0001)
X_hat = X.ix[:, feat_idx]


x = range(len(X_hat.columns))
labels = X_hat.columns
n_samples = len(X_hat)
for i in range(n_samples):
    if Y[i] ==1:
        plt.scatter(x,X_hat.iloc[i,],color='red',s=5)
    else:
        plt.scatter(x, X_hat.iloc[i,], color='green',s=5)
plt.show()
