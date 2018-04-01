import gcforest.caForest as ca
import gcforest.data_load as load
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold

X, Y = load.obesity_data()
cv = StratifiedKFold(n_splits=10, shuffle=False)

mean_fpr = np.linspace(0, 1, 100)
tprs = []
aucs = []

gcf = ca.gcForest(tolerance=0.0, min_samples_cascade=10)
f, ax = plt.subplots(1, 1)

for train, test in cv.split(X, Y):
    X_tr = X.iloc[train]
    y_tr = Y[train]

    X_te = X.iloc[test]
    y_te = Y[test]

    gcf.cascade_forest(X_tr, y_tr)

    pred_proba = gcf.cascade_forest(X_te)
    tmp = np.mean(pred_proba, axis=0)
    fpr, tpr, thresholds = roc_curve(Y[test], tmp[:, 1])
    v = interp(mean_fpr, fpr, tpr)
    tprs.append(v)
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='r', label='{}' '(auc = {:.3f})'.format('caForest', mean_auc), lw=2,
        alpha=.8)

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', alpha=.8)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc='lower right')
plt.show()
