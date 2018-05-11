# -*- coding:utf-8 -*-
import math
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC

import gcforest.data_load_phy as load2
import gcforest.data_load as load
from gcforest.gcforest import GCForest
from gcforest.utils.log_utils import get_logger
import json
import pandas as pd
LOGGER = get_logger('cascade_clf.lib.plot_roc_all')

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


def load_json(path):
    import json
    """
    支持以//开头的注释
    """
    lines = []
    with open(path) as f:
        for row in f.readlines():
            if row.strip().startswith("//"):
                continue
            lines.append(row)
    return json.loads("\n".join(lines))

def feat_indx(database_name, threhold):
    output_dir = osp.join("output", "result")
    path = osp.join(output_dir, database_name)
    file = open(path, 'r')
    dicts = json.load(file)

    for key, value in dicts.iteritems():
        if key == str(threhold):
            df = pd.DataFrame({'feature': value.keys(), 'importance': value.values()})
            df = df.sort_values(by=['importance'],ascending=False)

            feat_idx = df['feature'].tolist()

            feat_idx = [int(f) for f in feat_idx]

            # if database_name == 'obesity':
            #     return feat_idx[:250]
            # elif database_name == 't2d':
            #     return feat_idx[:50]
            return feat_idx

def cnn_acc(X, Y, train, test):
    ####CNN####################################################
    L1 = 32  # number of convolutions for first layer
    L2 = 64  # number of convolutions for second layer
    L3 = 1024  # number of neurons for dense layer
    learning_date = 1e-4  # learning rate
    epochs = 100  # number of times we loop through training data
    batch_size = 10  # number of data per batch

    Y_trans = one_hot(Y)
    train_data, test_data, train_labels, test_labels = X.iloc[train], X.iloc[test], Y_trans[train], \
                                                       Y_trans[test]
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
        print(
            "Epoch:", '%04d' % (epoch), "loss={:.9f}".format(avg_cost),
            "accuracy={:.9f}".format(avg_acc))
    ### cnn test###
    accuracy = accuracy.eval(feed_dict={xs: test_data, ys: test_labels, keep_prob: 1.0})
    print("conv_net accuracy = " + str(accuracy))
    y_pred = y_conv.eval(feed_dict={xs: test_data, ys: test_labels, keep_prob: 1.0})


    sess.close()

    return accuracy
    ###########################################################

def clf_acc(clf, X, Y, train, test):
    probas_ = None
    if isinstance(clf, GCForest):
        gc = clf
        x_train = X.iloc[train]
        y_train = Y[train]

        x_test = X.iloc[test]
        y_test = Y[test]

        X_train = x_train.values.reshape(-1, 1, len(x_train.columns))
        X_test = x_test.values.reshape(-1, 1, len(x_test.columns))

        X_train_enc = gc.fit_transform(X_train, y_train)

        probas_ = gc.predict(X_test)
        acc = accuracy_score(y_test, probas_)
        return acc

def autolabel(rects, ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height,
                "{:.3f}".format(height),
                ha='center', va='bottom',size=8)

def get_reduced_features():
    datasets = ['cirrhosis', 't2d', 'obesity']
    feat_len = []
    for dataset_idx,name in enumerate(datasets):
        length = len(feat_indx(name, 0.001))
        feat_len.append(length)
    return feat_len


if __name__ == "__main__":

    print get_reduced_features()
    save_fig = True
    # # ==============================================

    f, ax = plt.subplots(2, 2,figsize=(10,5))

    cv = StratifiedKFold(n_splits=5, shuffle=False, random_state=0)

    clf_rf = RandomForestClassifier(n_estimators=100, random_state=0)

    clf_svm = SVC(kernel='linear', C=1,
                  gamma=0.001, random_state=0, probability=True)

    config = load_json("/home/qiang/repo/python/cascade_clf/examples/demo_ca.json")
    clf_gc = GCForest(config)

    datasets = ['cirrhosis', 't2d', 'obesity']
    classifiers = [(clf_svm, 'black', "SVM"),
              (clf_rf, 'green', "Random Forest"),
              ('cnn', 'purple', "CNN"),
              (clf_gc, 'red', "Deep Forest")]

    for idx, classifier in enumerate(classifiers):
        acc_before = []
        acc_after = []
        for dataset_idx, name in enumerate(datasets):
            X = None
            Y = None
            if dataset_idx == 0:
                X, Y = load.cirrhosis_data()
                feat_idx = feat_indx(name, 0.001)
                X_hat = X.ix[:, feat_idx]
            elif dataset_idx == 1:
                X, Y = load.t2d_data()
                feat_idx = feat_indx(name, 0.001)
                X_hat = X.ix[:, feat_idx]
            elif dataset_idx == 2:
                X, Y = load.obesity_data()
                feat_idx = feat_indx(name, 0.001)
                X_hat = X.ix[:, feat_idx]
            else:
                raise Exception('the dataset is not defined!!!')

            if idx == 0 or idx ==1:
                clf_acc_before = cross_val_score(classifier[0], X, Y, cv=cv, scoring='accuracy')
                clf_acc_after = cross_val_score(classifier[0], X_hat, Y, cv=cv, scoring='accuracy')
            else:
                clf_acc_before = []
                clf_acc_after = []

                # if idx == 3:
                #     cv = StratifiedKFold(n_splits=5, shuffle=False, random_state=0)

                for train, test in cv.split(X, Y):
                    if idx == 2:  ## CNN
                        accuracy = cnn_acc(X, Y, train, test)
                        accuracy2 = cnn_acc(X_hat, Y, train, test)
                    elif idx == 3: ## gcForest
                        accuracy = clf_acc(classifier[0],X, Y, train, test)
                        accuracy2 = clf_acc(classifier[0],X_hat, Y, train, test)
                    clf_acc_before.append(accuracy)
                    clf_acc_after.append(accuracy2)

            acc_before.append(np.average(clf_acc_before))
            acc_after.append(np.average(clf_acc_after))

        n_groups = 3
        index = np.arange(n_groups)
        bar_width = 0.24
        opacity = 0.8


        if idx == 0:
            x, y = 0, 0
            color_a = 'm'
            color_b = 'y'
        elif idx == 1:
            x, y = 0, 1
            color_a = 'black'
            color_b = 'gray'
        elif idx == 2:
            x, y = 1, 0
            color_a = 'b'
            color_b = 'c'
        elif idx == 3:
            x, y = 1, 1
            color_a = 'r'
            color_b = 'g'
        ax[x, y].set_ylabel('Accuracy')
        ax[x, y].set_xticks(np.arange(n_groups) + bar_width / 2)
        ax[x, y].set_xticklabels(('cirrhosis', 't2d', 'obesity'))
        ax[x, y].set_ylim([0, 1.0])
        ax[x, y].set_title(classifier[2])


        rect1 = ax[x, y].bar(index, acc_before, bar_width,
                       alpha=opacity,
                       color=color_a,
                       label='Before')
        rect2 = ax[x, y].bar(index + bar_width, acc_after, bar_width,
                       alpha=opacity,
                       color=color_b,
                       label='After')

        autolabel(rect1, ax[x, y])
        autolabel(rect2, ax[x, y])
        ax[x, y].legend((rect1, rect2), ('Before', 'After'),loc='upper right', fontsize=8)

    plt.subplots_adjust(hspace=0.5)
    if save_fig:
        plt.savefig('fs.png', bbox_inches='tight')
        plt.close(f)
    else:
        plt.show()
