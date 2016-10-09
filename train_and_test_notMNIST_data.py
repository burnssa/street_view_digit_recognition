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
from six.moves import cPickle as pickle

pickle_file = 'notMNIST.pickle'

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

image_size = 28
IMAGE_PIXELS = image_size * image_size
num_labels = 10

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, IMAGE_PIXELS)).astype(np.float32)
    # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print("****After reformatting****")
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# Hidden layer parameters
beta = 0.0005
hlayer1_nodes = 1000
hlayer2_nodes = 200
hlayer3_nodes = 50
batch_size = 200
num_steps = 12000

# Learning rate decay parameters
decay_rate = 0.96
initial_learning_rate = 0.5

# Dropout parameters
keep_prob = 0.9

graph = tf.Graph()
with graph.as_default():
    # Big hat tip to - https://goo.gl/6KHb7n
    tf_train_dataset = tf.placeholder(tf.float32,
                                shape=(batch_size, IMAGE_PIXELS))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    weights = {
        'h1': tf.Variable(tf.truncated_normal([IMAGE_PIXELS, hlayer1_nodes], stddev = 0.01)),
        'h2': tf.Variable(tf.truncated_normal([hlayer1_nodes, hlayer2_nodes], stddev = 0.01)),
        'h3': tf.Variable(tf.truncated_normal([hlayer2_nodes, hlayer3_nodes], stddev = 0.01)),
        'out': tf.Variable(tf.truncated_normal([hlayer3_nodes, num_labels], stddev = 0.01))
    }
    biases = {
        'b1': tf.Variable(tf.truncated_normal([hlayer1_nodes])),
        'b2': tf.Variable(tf.truncated_normal([hlayer2_nodes])),
        'b3': tf.Variable(tf.truncated_normal([hlayer3_nodes])),
        'out': tf.Variable(tf.truncated_normal([num_labels]))
    }

    def perceptron(data_set, weights, biases):
        layer_1 = tf.nn.relu(tf.matmul(data_set, weights['h1']) + biases['b1'])
        if training_set(data_set):
            layer_1 = tf.nn.dropout(layer_1, keep_prob)
        layer_2 = tf.nn.relu(tf.matmul(layer_1, weights['h2']) + biases['b2'])
        if training_set(data_set):
            layer_2 = tf.nn.dropout(layer_2, keep_prob)
        layer_3 = tf.nn.relu(tf.matmul(layer_2, weights['h3']) + biases['b3'])
        if training_set(data_set):
            layer_3 = tf.nn.dropout(layer_3, keep_prob)
        out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
        return out_layer

    def training_set(data_set):
        return data_set == tf_train_dataset

    logits = perceptron(tf_train_dataset, weights, biases)

    l2_loss = 0
    for _, weight in weights.items():
        l2_loss = l2_loss + tf.nn.l2_loss(weight)

    for _, bias in biases.items():
        l2_loss = l2_loss + tf.nn.l2_loss(bias)

    pdb.set_trace()
    loss = (tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) +
        beta * l2_loss
    )
    print(loss)

    global_step = tf.Variable(0)  # count the number of steps taken.

    learning_rate = tf.train.exponential_decay(
        initial_learning_rate, global_step, num_steps, decay_rate
    )
    optimizer = (tf.train.GradientDescentOptimizer(
        learning_rate).minimize(
        loss, global_step=global_step)
    )

    train_prediction = tf.nn.softmax(logits)

    valid_tensor = perceptron(tf_valid_dataset, weights, biases)
    test_tensor = perceptron(tf_test_dataset, weights, biases)

    valid_prediction = tf.nn.softmax(valid_tensor)
    test_prediction = tf.nn.softmax(test_tensor)

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
        / predictions.shape[0])

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict
        )
        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print(l)
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(
                valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
