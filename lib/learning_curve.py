from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,StratifiedKFold,KFold

import gcforest.data_load_phy as load2
import numpy as np

if __name__ == "__main__":
    X, Y = load2.cirrhosis_data()
    x_tr, x_te, y_tr, y_te = train_test_split(X, Y,random_state=0,stratify=Y,test_size=0.2)


    n_trees = range(10, 100, 5)
    avg_accuracy = []
    for i in range(len(n_trees)):
        rf = RandomForestClassifier(n_trees[i], random_state=0)
        rf.fit(x_tr, y_tr)
        y_pred = rf.predict(x_te)
        acc = accuracy_score(y_te, y_pred)
        avg_accuracy.append(acc)
    print avg_accuracy
    print np.argmax(avg_accuracy)

