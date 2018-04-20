import gcforest.data_load_phy as load2
import gcforest.data_load as load
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

if __name__ == "__main__":
    X, Y = load2.cirrhosis_data()
    cv = StratifiedKFold(n_splits=10, shuffle=False, random_state=0)

    n_trees = [100,120,150,180]
    avg_accuracy = []
    for i in range(len(n_trees)):
        rf = RandomForestClassifier(n_trees[i],random_state=0)
        acc = []
        for train, test in cv.split(X, Y):
            X_tr = X.iloc[train]
            y_tr = Y[train]

            X_te = X.iloc[test]
            y_te = Y[test]
            rf.fit(X_tr, y_tr)
            y_pred = rf.predict(X_te)
            score = accuracy_score(y_te, y_pred)
            acc.append(score)
        avg_accuracy.append(np.mean(acc))

    x = range(len(n_trees))
    plt.plot(x, avg_accuracy, marker='o', mec='r')
    plt.legend()
    plt.xticks(x, n_trees)
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel("n_trees")
    plt.ylabel("accuracy")
    plt.title("Learning Curve")

    plt.show()
