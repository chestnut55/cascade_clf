import gcforest.caForest as ca
import gcforest.data_load_phy as load2
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X, Y = load2.cirrhosis_data()
cv = StratifiedKFold(n_splits=5, shuffle=False,random_state=42)


f, ax = plt.subplots(1, 1)
for i in range(2):
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    acc = []

    _name = None
    color = None
    for train, test in cv.split(X, Y):
        X_tr = X.iloc[train]
        y_tr = Y[train]

        X_te = X.iloc[test]
        y_te = Y[test]

        _pros = None
        y_pred = None
        if i == 0:
            gcf = ca.gcForest(tolerance=0.0,n_cascadeRFtree=50)
            gcf.cascade_forest(X_tr, y_tr)
            pred_proba = gcf.cascade_forest(X_te)
            _pros = np.mean(pred_proba, axis=0)
            color = 'g'
            y_pred = np.argmax(_pros, axis=1)
            _name = 'cascade forest'
        elif i==1:
            rf = RandomForestClassifier(n_estimators=50,random_state=42)
            rf.fit(X_tr, y_tr)
            _pros = rf.predict_proba(X_te)
            color = 'r'
            y_pred = rf.predict(X_te)
            _name = 'random forest'
        fpr, tpr, thresholds = roc_curve(Y[test], _pros[:, 1])
        v = interp(mean_fpr, fpr, tpr)
        tprs.append(v)
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        acc.append(accuracy_score(y_te, y_pred))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color=color, label='{}' '(auc = {:.3f})'.format(_name, mean_auc), lw=2,
            alpha=.8)

    print(_name + " accuracy=" + str(np.mean(acc)))
    print(_name + " auc=" + str(np.mean(aucs)))

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', alpha=.8)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc='lower right')
plt.show()
