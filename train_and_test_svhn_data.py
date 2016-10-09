### Deep learning capstone project
### Training and testing NotMNIST data for letter classification

# Import standard modeling modules

from __future__ import print_function
import numpy as np
import tensorflow as tf

# Modules for file extraction, plotting and storage

import matplotlib.pyplot as plt
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import pdb

pickle_file = 'SVHN_multi.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

# FIXME: determine whether this is optimal image size
image_size = 32
# Numbers 0 to 9, plus one for a blank number
num_labels = 11
# Includes classifier for length of the sequence and one for each of 5 digits
num_classifiers = 6
num_channels = 1

batch_size = 100
num_steps = 100000

# Convolutional parameters
patch_size = 4
depth1 = 16
depth2 = 32
depth3 = 64
conv_strides = [1,1,1,1]
ksize = [1,2,2,1]
max_pool_strides = [1,2,2,1]
lrn_alpha = 0.1
lrn_beta = 0.75
# lrn parameters chosen based on https://goo.gl/MUclw8 but alpha updated based on model performance

# Hidden layer parameters
beta = 0.005
hlayer1_nodes = 128
hlayer2_nodes = 64
hlayer3_nodes = 16

# Learning rate parameters
decay_rate = 0.98
initial_learning_rate = 0.05

# Dropout parameters
keep_prob = 0.8

graph = tf.Graph()
with graph.as_default():
    # Big hat tip to - https://goo.gl/6KHb7n
    tf_train_dataset = tf.placeholder(
        tf.float32, shape = ([batch_size, image_size, image_size, num_channels])
    )
    tf_train_labels = tf.placeholder(
        tf.int32, shape = ([batch_size, num_classifiers])
    )
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    def classifier_matrix():
        return np.arange(0, num_classifiers).tolist()

    weights = classifier_matrix()
    biases = classifier_matrix()

    conv_weights = {
        'cw1': tf.get_variable(
            shape = [patch_size, patch_size, num_channels, depth1],
            initializer = tf.contrib.layers.xavier_initializer_conv2d(),
            name='cw1'
        ),
        'cw2': tf.get_variable(
            shape = [patch_size, patch_size, depth1, depth2],
            initializer = tf.contrib.layers.xavier_initializer_conv2d(),
            name='cw2'
        ),
        'cw3': tf.get_variable(
            shape = [patch_size, patch_size, depth2, depth3],
            initializer = tf.contrib.layers.xavier_initializer_conv2d(),
            name='cw3'
        ),
    }

    conv_biases = {
        'cb1': tf.constant(1.0, shape = [depth1], name='cb1'),
        'cb2': tf.constant(1.0, shape = [depth2], name='cb2'),
        'cb3': tf.constant(1.0, shape = [depth3], name='cb3'),
    }

    def conv2d(data, weights):
        return tf.nn.conv2d(data, weights, conv_strides, padding = 'VALID')

    def max_pool(data):
        return tf.nn.max_pool(data, ksize, max_pool_strides, padding = 'SAME')

    def lrn(pooled_data):
        return tf.nn.local_response_normalization(
            pooled_data,
            alpha = lrn_alpha,
            beta = lrn_beta
        )

    def conv_layers(dataset):
        conv1 = tf.nn.relu(
            conv2d(dataset, conv_weights['cw1']) + conv_biases['cb1']
        )
        pool1 = max_pool(conv1)
        normalized1 = lrn(pool1)
        conv2 = tf.nn.relu(
            conv2d(normalized1, conv_weights['cw2']) + conv_biases['cb2']
        )
        pool2 = max_pool(conv2)
        normalized2 = lrn(pool2)
        conv3 = tf.nn.relu(
            conv2d(normalized2, conv_weights['cw3']) + conv_biases['cb3']
        )
        shape = conv3.get_shape().as_list()
        reshape_dim = [shape[0], shape[1] * shape[2] * shape[3]]
        reshape = tf.reshape(conv3, reshape_dim)
        reshape_cols = reshape_dim[1]
        return reshape, reshape_cols

    train_conv_output, reshape_cols = conv_layers(tf_train_dataset)

    for i in classifier_matrix():
        with tf.variable_scope("classifier_{}".format(i+1)) as scope:
            weights[i] =  {
                'h1': tf.get_variable(
                    shape = [reshape_cols, hlayer1_nodes],
                    initializer = tf.contrib.layers.xavier_initializer(),
                    name='h1'
                ),
                'h2': tf.get_variable(
                    shape = [hlayer1_nodes, hlayer2_nodes], # hlayer2_nodes
                    initializer = tf.contrib.layers.xavier_initializer(),
                    name='h2'
                ),
                'h3': tf.get_variable(
                    shape = [hlayer2_nodes, hlayer3_nodes],
                    initializer = tf.contrib.layers.xavier_initializer(),
                    name='h3'
                ),
                'out': tf.get_variable(
                    shape = [hlayer3_nodes, num_labels],
                    initializer = tf.contrib.layers.xavier_initializer(),
                    name='out'
                )
            }
            biases[i] = {
                'b1': tf.Variable(
                    tf.truncated_normal([hlayer1_nodes], stddev=0.1)
                ),
                'b2': tf.Variable(
                    tf.truncated_normal([hlayer2_nodes], stddev=0.1)
                ),
                'b3': tf.Variable(
                    tf.truncated_normal([hlayer3_nodes], stddev=0.1)
                ),
                'out': tf.Variable(
                    tf.truncated_normal([num_labels], stddev=0.1)
                )
            }

    def perceptron(data_set, weights, biases):
        layer_1, layer_2, layer_3, out_layer = classifier_matrix(), \
            classifier_matrix(), classifier_matrix(), classifier_matrix()
        for i in classifier_matrix():
            layer_1[i] = tf.nn.relu(
                tf.matmul(data_set, weights[i]['h1']) + biases[i]['b1']
            )
            if training_set(data_set):
                layer_1[i] = tf.nn.dropout(layer_1[i], keep_prob)
            layer_2[i] = tf.nn.relu(
                tf.matmul(layer_1[i], weights[i]['h2']) + biases[i]['b2']
            )
            if training_set(data_set):
                layer_2[i] = tf.nn.dropout(layer_2[i], keep_prob)
            layer_3[i] = tf.nn.relu(
                tf.matmul(layer_2[i], weights[i]['h3']) + biases[i]['b3']
            )
            if training_set(data_set):
                layer_3[i] = tf.nn.dropout(layer_3[i], keep_prob)
            out_layer[i] = tf.matmul(
                layer_3[i], weights[i]['out']
            ) + biases[i]['out']
        return out_layer

    def training_set(data_set):
        return data_set == train_conv_output

    def prediction(logits):
        return tf.pack([tf.nn.softmax(logits[i]) for i in classifier_matrix()])

    def total_loss(weights, biases, logits, labels):
        with tf.variable_scope('loss') as scope:
            sum_loss = sum(
                tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits[i], labels[:,i]
                    )
                ) for i in classifier_matrix()
            )

        l2_loss = 0
        for i in classifier_matrix():
            for _, weight in weights[i].items():
                l2_loss = l2_loss + tf.nn.l2_loss(weight[i, :])

            for _, bias in biases[i].items():
                l2_loss = l2_loss + tf.nn.l2_loss(bias[i])

        additive_l2_loss = l2_loss * beta
        return sum_loss + additive_l2_loss

    logits = perceptron(train_conv_output, weights, biases)

    global_step = tf.Variable(0)  # count the number of steps taken.

    learning_rate = tf.train.exponential_decay(
        initial_learning_rate, global_step, num_steps, decay_rate
    )
    loss = total_loss(weights, biases, logits, tf_train_labels)

    optimizer = (tf.train.GradientDescentOptimizer(learning_rate)
        .minimize(loss, global_step=global_step)
    )

    train_prediction = prediction(logits)

    valid_conv_output, _ = conv_layers(tf_valid_dataset)
    test_conv_output, _ = conv_layers(tf_test_dataset)

    valid_tensor = perceptron(valid_conv_output, weights, biases)
    test_tensor = perceptron(test_conv_output, weights, biases)

    valid_prediction = prediction(valid_tensor)
    test_prediction = prediction(test_tensor)

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 2).T == labels) /
        predictions.shape[1] /
        predictions.shape[0]
    )

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized")
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {
            tf_train_dataset : batch_data,
            tf_train_labels : batch_labels
        }

        _, final_loss, predictions = session.run(
            [optimizer, loss, train_prediction],
            feed_dict=feed_dict
        )

        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, final_loss))
            print("Minibatch accuracy: %.1f%%" % accuracy(
                predictions, batch_labels)
            )
            print("Validation accuracy: %.1f%%" % accuracy(
                valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(
        test_prediction.eval(), test_labels)
    )
