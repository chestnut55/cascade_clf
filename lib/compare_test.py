from modified_stack_forest import StackingCascadeForest
from deep_forest import CascadeForest
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
import gcforest.data_load as load
import gcforest.data_load_phy as load2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def get_stacking_est_config():
    stacking_estimators_config = {
        'cascade': [
            {
                'estimator_class': RandomForestClassifier,
                'estimator_params': {
                    'n_estimators': 500,
                    'n_jobs': -1,
                    'random_state': 0
                }
            },
            {
                'estimator_class': ExtraTreesClassifier,
                'estimator_params': {
                    'n_estimators': 500,
                    'n_jobs': -1,
                    'max_features': 1,
                    'random_state': 0
                }
            },{
                'estimator_class': RandomForestClassifier,
                'estimator_params': {
                    'n_estimators': 500,
                    'n_jobs': -1,
                    'random_state': 0
                }
            },
            {
                'estimator_class': ExtraTreesClassifier,
                'estimator_params': {
                    'n_estimators': 500,
                    'n_jobs': -1,
                    'max_features': 1,
                    'random_state': 0
                }
            }
        ]
    }

    return stacking_estimators_config

def get_cascade_est_config():
    estimators_config = {
        'cascade': [{
            'estimator_class': RandomForestClassifier,
            'estimator_params': {
                'n_estimators': 500,
                'n_jobs': -1,
                'random_state': 0
            }
        },
            {
                'estimator_class': ExtraTreesClassifier,
                'estimator_params': {
                    'n_estimators': 500,
                    'n_jobs': -1,
                    'random_state': 0,
                    'max_features': 1
                }
            },
            {
                'estimator_class': RandomForestClassifier,
                'estimator_params': {
                    'n_estimators': 500,
                    'n_jobs': -1,
                    'random_state': 0
                }
            },
            {
                'estimator_class': ExtraTreesClassifier,
                'estimator_params': {
                    'n_estimators': 500,
                    'n_jobs': -1,
                    'random_state': 0,
                    'max_features': 1
                }
            },
            {
                'estimator_class': RandomForestClassifier,
                'estimator_params': {
                    'n_estimators': 500,
                    'n_jobs': -1,
                    'random_state': 0
                }
            },
            {
                'estimator_class': ExtraTreesClassifier,
                'estimator_params': {
                    'n_estimators': 500,
                    'n_jobs': -1,
                    'random_state': 0,
                    'max_features': 1
                }
            },
            {
                'estimator_class': RandomForestClassifier,
                'estimator_params': {
                    'n_estimators': 500,
                    'n_jobs': -1,
                    'random_state': 0
                }
            },
            {
                'estimator_class': ExtraTreesClassifier,
                'estimator_params': {
                    'n_estimators': 500,
                    'n_jobs': -1,
                    'random_state': 0,
                    'max_features': 1
                }
            }
        ]
    }

    return estimators_config

acc1 = []
acc2 = []
acc3 = []
for i in range(10):
    X, Y = load2.yeast_data()
    x_tr,x_te,y_tr,y_te = train_test_split(X,Y,stratify=Y,test_size=0.3)

    # cascade forest
    estimators_config = get_cascade_est_config()
    c_forest = CascadeForest(estimators_config['cascade'])
    c_forest.fit(x_tr, y_tr)
    y_pred = c_forest.predict(x_te)
    accuracy1 = accuracy_score(y_te, y_pred)
    acc1.append(accuracy1)
    print(accuracy1)

    # stacking cascade forest
    stacking_estimators_config = get_stacking_est_config()
    s_forest = StackingCascadeForest(stacking_estimators_config['cascade'])
    s_forest.fit(x_tr, y_tr)
    y_pred = s_forest.predict(x_te)
    accuracy2 = accuracy_score(y_te, y_pred)
    acc2.append(accuracy2)
    print(accuracy2)

    rf = RandomForestClassifier(n_estimators=500, random_state=0)
    rf.fit(x_tr, y_tr)
    y_pred = rf.predict(x_te)
    accuracy3 = accuracy_score(y_te, y_pred)
    acc3.append(accuracy3)
    print(accuracy3)

print(acc1)
print(acc2)
print(acc3)

