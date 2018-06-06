import numpy as np
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import pdist, cdist
import logging
import uuid
import gcforest.data_load as load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def create_logger(instance, verbose):
    logger = logging.getLogger(str(uuid.uuid4()))
    fmt = logging.Formatter('{} - %(message)s'.format(instance))
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    return logger


def modified_FW(P, Q, lamda, tao, S, n_estimators):
    """
    learning the tree weight for each forest
    :param P:  is a vector
            Euclidean distance of class distribution for the two samples i and j in a forest
    :param Q: is a vector
            L1 distance of class distribution for the two samples i and j in a forest
    :param z: 1 is two sample is different, 0 otherwise
    :param lamda: the parameter for regulization of W
    :param tao: the margin for the simimarity and dissimilarity
    :param S: number of iterations
    :param n_estimators: # the number of trees in a forest
    :return: the tree weight in the forest
    """
    # initial estimate, could be any feasible point
    # w_t = np.eye(1, len(n_estimators), 0)  # the tree weight in the forest
    w_t = np.repeat(1.0/len(n_estimators), len(n_estimators))
    w_t = np.asarray(w_t).flatten()


    for k in range(S):
        Jw = []
        s_i_j = tao - np.dot(np.asarray(Q), w_t)
        for (i, est) in enumerate(n_estimators):
            # compute the gradient of loss function J(w)
            grad_Jw = 2 * w_t[i] * (lamda + P[i])
            if s_i_j > 0:
                grad_Jw -= 2 * s_i_j * s_i_j * Q[i]
            Jw.append(grad_Jw)
        min_idx = np.argmin(np.asarray(Jw))
        g_s = np.zeros(len(n_estimators))
        g_s[min_idx] = 1
        step_size = 2.0 / (k + 2.0)
        w_t = w_t + step_size * (g_s - w_t)
    return w_t


def calculate_euclidean_l1_distance(X_proba, y):
    indexes_0 = [i for i, x in enumerate(y) if x == 0]
    indexes_1 = [i for i, x in enumerate(y) if x == 1]
    # Euclidean distance for two samples i and j in a tree belongs to the same class
    p_i_j = np.sum(pdist(X_proba[indexes_0,], 'euclidean')) + np.sum(pdist(X_proba[indexes_1,], 'euclidean'))
    # L1 distance for two samples i and j in a tree belongs to the different classes
    q_i_j = np.sum(cdist(X_proba[indexes_0,], X_proba[indexes_1,], 'minkowski', p=1))

    return [p_i_j, q_i_j]


def DDF(X, y, forest):
    """
    :param X_proba: (nsamples,class_distribution) e.g (0.6,0.4)
    :param y:(nsamples,labels) e.g 1
    :return: the weight of decision tree in the forest
    """
    lamda = -100
    tao = 10
    S = 100

    P = []
    Q = []
    for tree in forest.estimators_:
        X_proba = tree.predict_proba(X)
        [p, q] = calculate_euclidean_l1_distance(X_proba, y)
        P.append(p)
        Q.append(q)

    return modified_FW(np.asarray(P), np.asarray(Q), lamda, tao, S, forest.estimators_)


class CascadeForest():
    """
    CascadeForest
    @param estimators_config    A list containing the class and parameters of the estimators for
                                the CascadeForest.
    @param folds                The number of k-folds to use.
    @param verbose              Adds verbosity.
    """

    def __init__(self, estimators_config, folds=3, verbose=False):
        self.estimators_config = estimators_config
        self.folds = folds

        self.logger = create_logger(self, verbose)

    def predict_probability(self, estimator, x_tr, weight):
        """
        predict the probability for the estimator with the weighted decision tree
        :param estimator: the Forest estimator
        :return:
        """
        sample_distribution = []
        for (i, tree_in_rf) in enumerate(estimator.estimators_):
            tr_samples_class_dist = tree_in_rf.predict_proba(x_tr) * weight[i]
            sample_distribution.append(tr_samples_class_dist)

        sample_distribution = np.asarray(sample_distribution)

        return np.sum(sample_distribution, axis=0)

    def fit(self, X, y):
        self.logger.info('Cascade fitting for X ({}) and y ({}) started'.format(X.shape, y.shape))
        self.classes = np.unique(y)
        self.level = 0
        self.levels = []
        self.weights = None
        self.max_score = None

        while True:
            self.logger.info('Level #{}:: X with shape: {}'.format(self.level + 1, X.shape))
            estimators = [
                estimator_config['estimator_class'](**estimator_config['estimator_params'])
                for estimator_config in self.estimators_config
            ]

            predictions = []
            weights = []
            for estimator in estimators:
                self.logger.debug('Fitting X ({}) and y ({}) with estimator {}'.format(
                    X.shape, y.shape, estimator
                ))
                estimator.fit(X, y)

                weight = DDF(X, y, estimator)  ## weight for the decision trees in the forest
                print weight
                weighted_prediction = self.predict_probability(estimator, X, weight)
                # TODO
                # TODO
                # TODO
                #
                # Gets a prediction of X with shape (len(X), n_classes)
                #
                # prediction = cross_val_predict(
                #     estimator,
                #     X,
                #     y,
                #     cv=self.folds,
                #     method='predict_proba',
                #     n_jobs=-1,
                # )

                predictions.append(weighted_prediction)

                weights.append(weight)

            self.logger.info('Level {}:: got all predictions'.format(self.level + 1))

            #
            # Stacks horizontally the predictions to each of the samples in X
            #
            X = np.hstack([X] + predictions)

            #
            # For each sample, compute the average of predictions of all the estimators, and take
            # the class with maximum score for each of them.
            #
            y_prediction = self.classes.take(
                np.array(predictions).mean(axis=0).argmax(axis=1)
            )

            score = accuracy_score(y, y_prediction)
            self.logger.info('Level {}:: got accuracy {}'.format(self.level + 1, score))
            if self.max_score is None or score > self.max_score:
                self.level += 1
                self.max_score = score
                self.levels.append(estimators)
                self.weights = weights
            else:
                break

    def predict(self, X):
        for estimators in self.levels:
            predictions = []
            for (i, estimator) in enumerate(estimators):
                w = np.asarray(self.weights[i])
                predictions.append(self.predict_probability(estimator, X, w))
            self.logger.info('Shape of predictions: {} shape of X: {}'.format(
                np.array(predictions).shape, X.shape
            ))
            X = np.hstack([X] + predictions)

        return self.classes.take(
            np.array(predictions).mean(axis=0).argmax(axis=1)
        )

    def __repr__(self):
        return '<CascadeForest forests={}>'.format(len(self.estimators_config))


if __name__ == "__main__":
    estimators_config = {
        'cascade': [{
            'estimator_class': RandomForestClassifier,
            'estimator_params': {
                'n_estimators': 50,
                'n_jobs': -1,
                'random_state': 42,
                'min_samples_leaf': 10
            }
        }, {
            'estimator_class': RandomForestClassifier,
            'estimator_params': {
                'n_estimators': 50,
                'max_features': 1,
                'n_jobs': -1,
                'random_state': 42,
                'min_samples_leaf': 10
            }
        }]
    }

    X, Y = load.cirrhosis_data()

    X_tr, X_te, y_tr, y_te = train_test_split(X, Y)

    c_forest = CascadeForest(estimators_config['cascade'])
    c_forest.fit(X_tr, y_tr)

    y_pred = c_forest.predict(X_te)
    accuracy = accuracy_score(y_te, y_pred)
    print("predict accuracy=", accuracy)

    rf = RandomForestClassifier(n_estimators=50)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    accuracy = accuracy_score(y_te, y_pred)
    print("random forest predict accuracy=", accuracy)
