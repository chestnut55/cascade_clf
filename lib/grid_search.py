from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

import gcforest.data_load as load
import gcforest.data_load_phy as load2

params = {"n_estimators": [20, 50, 80, 100, 150, 200, 300],
          "max_depth": [3, 4, 6, 10],
          "min_samples_split": [2, 5, 10]
          }
rf = RandomForestClassifier()
grid_search = GridSearchCV(estimator=rf, param_grid=params, n_jobs=-1, cv=5)

X, Y = load2.t2d_data()

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=0)
# grid_search.fit(X_train, y_train)
#
# print("=====================")
# print(grid_search.best_score_)
# print(grid_search.best_params_)

cv = StratifiedKFold(n_splits=5, shuffle=False, random_state=0)
for train, test in cv.split(X, Y):
    grid_search.fit(X.iloc[train], Y[train])

    print("=====================")
    print(grid_search.best_score_)
    print(grid_search.best_params_)
