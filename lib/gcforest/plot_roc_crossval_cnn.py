import math

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

import lib.gcforest.data_load as load
from lib.gcforest.gcforest import GCForest


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def one_hot(integer_encoded):
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
    return one_hot_encoded

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
    # ca_config["estimators"].append({"n_folds": 5, "type": "LogisticRegression"})
    config["cascade"] = ca_config
    return config


if __name__ == "__main__":
    X, Y = load.hmp_hmpii_data()
    cv = StratifiedKFold(n_splits=5, shuffle=False)

    config = gcforest_config()
    clf_gc = GCForest(config)
    gc_pred_acc = []

    cnn_pred_acc = []

    # # ==============================================
    f, ax = plt.subplots(1, 1)
    params = [(clf_gc, 'red', "Deep Forest"),
              ("cnn", 'blue',"CNN")]

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
                ####CNN####################################################
                L1 = 32  # number of convolutions for first layer
                L2 = 64  # number of convolutions for second layer
                L3 = 1024  # number of neurons for dense layer
                learning_date = 1e-4  # learning rate
                epochs = 50  # number of times we loop through training data
                batch_size = 10  # number of data per batch

                Y_trans = one_hot(Y)
                train_data, test_data, train_labels, test_labels = X.iloc[train], X.iloc[test], Y_trans[train], Y_trans[test]
                features = train_data.shape[1]
                classes = train_labels.shape[1]
                sess = tf.InteractiveSession()

                xs = tf.placeholder(tf.float32, [None, features])
                ys = tf.placeholder(tf.float32, [None, classes])
                keep_prob = tf.placeholder(tf.float32)
                x_shape = tf.reshape(xs, [-1, 1, features, 1])

                # first conv
                w_conv1 = weight_variable([5, 5, 1, L1])
                b_conv1 = bias_variable([L1])
                h_conv1 = tf.nn.relu(conv2d(x_shape, w_conv1) + b_conv1)
                h_pool1 = max_pool_2x2(h_conv1)

                # second conv
                w_conv2 = weight_variable([5, 5, L1, L2])
                b_conv2 = bias_variable([L2])
                h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
                h_pool2 = max_pool_2x2(h_conv2)

                tmp_shape = (int)(math.ceil(features / 4.0))
                h_pool2_flat = tf.reshape(h_pool2, [-1, 1 * tmp_shape * L2])

                # third dense layer,full connected
                w_fc1 = weight_variable([1 * tmp_shape * L2, L3])
                b_fc1 = bias_variable([L3])
                h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

                h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

                # fourth layer, output
                w_fc2 = weight_variable([L3, classes])
                b_fc2 = bias_variable([classes])
                y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

                cost = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(y_conv), reduction_indices=[1]))
                optimizer = tf.train.AdamOptimizer(learning_date).minimize(cost)

                correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(ys, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                init = tf.global_variables_initializer()
                sess.run(init)

                ### cnn start train##################################
                for epoch in range(epochs):
                    avg_cost = 0.
                    avg_acc = 0.
                    for batch in range(len(train_data) // batch_size):
                        offset = (batch * batch_size) % len(train_data)
                        batch_data = train_data[offset:(offset + batch_size)]
                        batch_labels = train_labels[offset:(offset + batch_size)]
                        _, c, acc = sess.run([optimizer, cost, accuracy],
                                             feed_dict={xs: batch_data, ys: batch_labels, keep_prob: 0.5})
                        avg_cost += c / (len(train_data) // batch_size)
                        avg_acc += acc / (len(train_data) // batch_size)
                    print("Epoch:", '%04d' % (epoch), "loss={:.9f}".format(avg_cost), "accuracy={:.9f}".format(avg_acc))
                ### cnn test###
                accuracy = accuracy.eval(feed_dict={xs: test_data, ys: test_labels, keep_prob: 1.0})
                print("conv_net accuracy = " + str(accuracy))
                cnn_pred_acc.append(accuracy)
                y_pred = y_conv.eval(feed_dict={xs: test_data, ys: test_labels, keep_prob: 1.0})

                print("y_pred=",y_pred.shape)
                fpr, tpr, thresholds = roc_curve(Y[test], y_pred[:, 1])
                v = interp(mean_fpr, fpr, tpr)
                tprs.append(v)
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)

                sess.close()
                ###########################################################

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color=x[1], label='{}' '(auc = {:.3f})'.format(x[2], mean_auc), lw=2,
                alpha=.8)

    print("Accuracy Deep Forest: %0.3f (+/- %0.3f)" % (np.array(gc_pred_acc).mean(), np.array(gc_pred_acc).std() * 2))
    print("Accuracy Deep Forest: %0.3f (+/- %0.3f)" % (np.array(cnn_pred_acc).mean(), np.array(cnn_pred_acc).std() * 2))

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', alpha=.8)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc='lower right')
    plt.show()
