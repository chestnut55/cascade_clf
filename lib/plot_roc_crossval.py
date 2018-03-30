import gcforest.data_load as load
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from gcforest.gcforest import GCForest
import numpy as np

def gcforest_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 100, "max_depth": None, "n_jobs": -1})
    # ca_config["estimators"].append(
    #         {"n_folds": 5, "type": "XGBClassifier", "n_estimators": 10, "max_depth": 5,
    #          "objective": "multi:softprob", "silent": True, "nthread": -1, "learning_rate": 0.1} )
    # ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier"})
    # ca_config["estimators"].append({"n_folds": 10, "type": "LogisticRegression","C":1.0,"penalty":"l1","tol":1e-4})
    config["cascade"] = ca_config
    return config

def print_importance(clf):
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("indices=", len(indices))

    # Print the feature ranking
    print("The last Classifier Feature ranking:")
    # features = X_train_enc.shape[1]
    num_selected_features = 30
    print("features=", num_selected_features)

    indices = indices[:num_selected_features]
    for f in range(0, num_selected_features):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

if __name__ == "__main__":
    X, Y = load.hmp_hmpii_data()


    cv = StratifiedKFold(n_splits=10,shuffle=False)

    clf_rf = RandomForestClassifier(
        n_estimators=100, random_state=0)
    scores = cross_val_score(clf_rf, X, Y, cv=cv, scoring='accuracy')
    print("clf_rf", scores)
    print("Accuracy of Random Forest Classifier: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

    clf_svm = SVC(kernel='linear', C=1,
                  gamma=0.001, random_state=0, probability=True)
    scores = cross_val_score(clf_svm, X, Y, cv=cv, scoring='accuracy')
    print("clf_svm", scores)
    print("Accuracy SVM Classifier: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))


    config = gcforest_config()
    clf_gc = GCForest(config)
    gc_pred_acc = []

    # # ==============================================
    f, ax = plt.subplots(1, 1)
    params = [(clf_rf, 'green', "Random Forest"),
              (clf_svm, 'black', "SVM"),
              (clf_gc,'red',"Deep Forest")]

    for x in params:
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []
        for train, test in cv.split(X, Y):
            if isinstance(x[0], GCForest):
                gc = x[0]
                x_train = X.iloc[train]
                y_train = Y[train]

                x_test = X.iloc[test]
                y_test = Y[test]

                x_train = x_train.values.reshape(-1, 1, len(x_train.columns)).astype('float32')
                x_test = x_test.values.reshape(-1, 1, len(x_test.columns)).astype('float32')

                X_train = x_train[:, np.newaxis, :, :]
                X_test = x_test[:, np.newaxis, :, :]

                X_train_enc = gc.fit_transform(X_train, y_train)
                # y_pred = gc.predict(X_test)
                # acc = accuracy_score(y_test, y_pred)
                # gc_pred_acc.append(acc)
                # print("Test Accuracy of GcForest = {:.2f} %".format(acc * 100))
                # probas_ = gc.predict_proba(X_test)
                # fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
                # v = interp(mean_fpr, fpr, tpr)
                # tprs.append(v)
                # tprs[-1][0] = 0.0
                # roc_auc = auc(fpr, tpr)
                # aucs.append(roc_auc)

                # You can try passing X_enc to another classfier on top of gcForest.e.g. xgboost/RF.
                X_test_enc = gc.transform(X_test)
                X_train_enc = X_train_enc.reshape((X_train_enc.shape[0], -1))
                X_test_enc = X_test_enc.reshape((X_test_enc.shape[0], -1))
                X_train_origin = X_train.reshape((X_train.shape[0], -1))
                X_test_origin = X_test.reshape((X_test.shape[0], -1))
                X_train_enc = np.hstack((X_train_origin, X_train_enc))
                X_test_enc = np.hstack((X_test_origin, X_test_enc))
                clf = RandomForestClassifier(n_estimators=1000, max_depth=None, n_jobs=-1)
                clf.fit(X_train_enc, y_train)

                ### print the important features
                print_importance(clf)

                y_pred = clf.predict(X_test_enc)
                acc = accuracy_score(y_test, y_pred)
                gc_pred_acc.append(acc)
                print("Test Accuracy of clf GcForest = {:.2f} %".format(acc * 100))
                probas_ = clf.predict_proba(X_test_enc)
                fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
                v = interp(mean_fpr, fpr, tpr)
                tprs.append(v)
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)

            else:
                probas_ = x[0].fit(X.iloc[train], Y[train]).predict_proba(X.iloc[test])
                fpr, tpr, thresholds = roc_curve(Y[test], probas_[:, 1])
                v = interp(mean_fpr, fpr, tpr)
                tprs.append(v)
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color=x[1], label='{}' '(auc = {:.3f})'.format(x[2], mean_auc), lw=2,
                alpha=.8)

    print("Accuracy Deep Forest: %0.3f (+/- %0.3f)" % (np.array(gc_pred_acc).mean(), np.array(gc_pred_acc).std() * 2))

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', alpha=.8)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc='lower right')
    plt.show()
