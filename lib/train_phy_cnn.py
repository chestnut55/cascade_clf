# -*- coding:utf-8 -*-
import math

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.preprocessing import OneHotEncoder
from joblib import Parallel, delayed
from numpy import unique
import argparse
import math
from gcforest.utils.config_utils import load_json
import multiprocessing
import lib.gcforest.data_load as load
from lib.gcforest.gcforest import GCForest
from gcforest.datasets.graph import Graph
import pandas as pd


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

def generate_maps(x, g, f):
    g.populate_graph(f, x)
    return x, np.array(g.get_map())

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", type=str, default=None, help="gcforest Net Model File")
    args = parser.parse_args()
    return args

L1 = 64  # number of convolutions for first layer
L2 = 64  # number of convolutions for second layer
L3 = 64  # number of neurons for dense layer
L4 = 1024
learning_date = 1e-4  # learning rate
epochs = 100  # number of times we loop through training data
batch_size = 10  # number of data per batch

data_name = "cirrhosis"
data_dir = "../lib/gcforest/data/" + data_name

my_x = []
my_y = []

my_x = np.loadtxt(data_dir + '/count_matrix.csv', dtype=np.float32, delimiter=',')

my_x = (my_x - my_x.min()) / (my_x.max() - my_x.min())

my_y = np.genfromtxt(data_dir + '/labels.txt', dtype=np.str_, delimiter=',')
features = np.genfromtxt(data_dir + '/otu.csv', dtype=np.str_, delimiter=',')

num_samples = my_x.shape[0]
num_features = len(my_x[0])
classes = list(unique(my_y))
num_classes = len(classes)

my_ref = pd.factorize(my_y)[1]
f = open(data_dir + "/label_reference.txt", 'w')
f.write(str(my_ref))
f.close()

g = Graph()
g.build_graph(data_dir + "/newick.txt")

my_data = pd.DataFrame(my_x)
my_data = np.array(my_data)
my_lab = pd.factorize(my_y)[0]
my_maps = []
my_benchmark = []

num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores)(delayed(generate_maps)(x, g, features) for x in my_data)
my_maps.append(np.array(np.take(results, 1, 1).tolist()))
my_benchmark.append(np.array(np.take(results, 0, 1).tolist()))

my_maps = np.array(my_maps)
my_benchmark = np.array(my_benchmark)
map_rows = my_maps.shape[2]
map_cols = my_maps.shape[3]

args = parse_args()
config = load_json(args.model)
gc = GCForest(config)
X = my_maps[0].reshape(-1, map_rows, map_cols)


train_data, test_data, train_labels, test_labels = train_test_split(X, my_lab, random_state=0, test_size=0.2, stratify=my_lab)

train_labels = one_hot(train_labels)
test_labels = one_hot(test_labels)
y_train = np.squeeze(np.array(train_labels).reshape(1, -1), 0)
y_test = np.squeeze(np.array(test_labels).reshape(1, -1), 0)


levels = train_data.shape[1]
features = train_data.shape[2]
classes = 2
sess = tf.InteractiveSession()

xs = tf.placeholder(tf.float32, [None,levels, features])
ys = tf.placeholder(tf.float32, [None, classes])
keep_prob = tf.placeholder(tf.float32)
x_shape = tf.reshape(xs, [-1, levels, features, 1])

# first conv
w_conv1 = weight_variable([5, 10, 1, L1])
b_conv1 = bias_variable([L1])
h_conv1 = tf.nn.relu(conv2d(x_shape, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second conv
w_conv2 = weight_variable([4, 10, L1, L2])
b_conv2 = bias_variable([L2])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# thrid conv
w_conv3 = weight_variable([3, 10, L2, L3])
b_conv3 = bias_variable([L3])
h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv2) + b_conv2)
h_pool3 = max_pool_2x2(h_conv3)

nrow = (int)(math.ceil(levels / 8.0))
ncol = (int)(math.ceil(features / 8.0))

h_pool3_flat = tf.reshape(h_pool3, [-1, nrow * ncol * L3])

# third dense layer,full connected
w_fc1 = weight_variable([nrow * ncol * L3, L4])
b_fc1 = bias_variable([L4])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# fourth layer, output
w_fc2 = weight_variable([L4, classes])
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