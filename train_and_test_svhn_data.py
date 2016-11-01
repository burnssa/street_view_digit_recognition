### Deep learning capstone project
### Training and testing SVHN data for letter classification

# Import standard modeling modules

from __future__ import print_function
import numpy as np
import tensorflow as tf

# Modules for file extraction, plotting and storage

import matplotlib.pyplot as plt
import os
import sys
import tarfile
from scipy import ndimage
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import pdb
import timeit
import collections
from scipy.misc import imresize
import datetime
import matplotlib.image as mpimg

pickle_file = 'SVHN_multi.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    real_life_dataset = save['real_life_dataset']
    del save  # hint to help gc free up memory

    # Final dataset cleanup
    # Remove element where listed as 6-digit number
    six_digit_index = np.where(train_labels[:,0] == 6)[0][0]
    train_labels = np.delete(train_labels, six_digit_index, 0)
    train_dataset = np.delete(train_dataset, six_digit_index, 0)

    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)
    print('Real life set', real_life_dataset.shape)

real_life_flag = False # False to run training / True to predict labels for real images

image_size = 32
# Numbers 0 to 9, plus 10 to represent a blank potential digit
num_labels = 11
# Includes classifier for length of the sequence and one for each of 5 digits
num_classifiers = 6
num_channels = 1

batch_size = 100
num_steps = 200000

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
# lrn parameters chosen based on https://goo.gl/MUclw8,
# but with alpha updated based on initial model performance

# Hidden layer parameters
hlayer1_nodes = 128
hlayer2_nodes = 64
hlayer3_nodes = 16

# Loss function parameters
beta = 0.0005
decay_rate = 0.98
initial_learning_rate = 0.05

# Dropout parameters
keep_prob = 0.8

# Varying layer parameters
potential_total_layers = 7

start_time = timeit.default_timer()
test_result_table = []

for num_layers in range(2, potential_total_layers + 1):
    lap_time_start = timeit.default_timer()
    tf.reset_default_graph()
    graph = tf.get_default_graph()
    with graph.as_default():
        # Big hat tip to - https://goo.gl/6KHb7n
        tf_train_dataset = tf.placeholder(
            tf.float32,
            shape = ([batch_size, image_size, image_size, num_channels])
        )
        tf_train_labels = tf.placeholder(
            tf.int32, shape = ([batch_size, num_classifiers])
        )
        tf_valid_dataset = tf.constant(valid_dataset)
        if real_life_flag:
            tf_test_dataset = tf.constant(real_life_dataset, dtype = tf.float32)
        else:
            tf_test_dataset = tf.constant(test_dataset)

        def classifier_matrix():
            return np.arange(0, num_classifiers).tolist()

        weights = classifier_matrix()
        biases = classifier_matrix()

        conv_shape1 = [patch_size, patch_size, num_channels, depth1]
        conv_shape2 = [patch_size, patch_size, depth1, depth2]
        conv_shape3 = [patch_size, patch_size, depth2, depth3]

        if num_layers <= 5:
            conv_shape3 = conv_shape2 = conv_shape1
            depth3 = depth2 = depth1
        elif num_layers == 6:
            conv_shape3 = conv_shape2
            conv_shape2 = conv_shape1
            depth3 = depth2
            depth2 = depth1
        else:
            pass

        conv_weights = {
            'cw1': tf.get_variable(
                shape = conv_shape1,
                initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                name='cw1'
            ),
            'cw2': tf.get_variable(
                shape = conv_shape2,
                initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                name='cw2'
            ),
            'cw3': tf.get_variable(
                shape = conv_shape3,
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
            return tf.nn.max_pool(
                data,
                ksize,
                max_pool_strides,
                padding = 'SAME'
            )

        def lrn(pooled_data):
            return tf.nn.local_response_normalization(
                pooled_data,
                alpha = lrn_alpha,
                beta = lrn_beta
            )

        def conv_layers(dataset):
            print("Num layers: {}".format(num_layers))
            conv1 = tf.nn.relu(
                conv2d(dataset, conv_weights['cw1']) + conv_biases['cb1']
            )
            pool1 = max_pool(conv1)

            if num_layers < 7:
                normalized1 = dataset
            else:
                normalized1 = lrn(pool1)

            conv2 = tf.nn.relu(
                conv2d(normalized1, conv_weights['cw2']) + conv_biases['cb2']
            )
            pool2 = max_pool(conv2)

            if num_layers < 6:
                normalized2 = normalized1
            else:
                normalized2 = lrn(pool2)

            conv3 = tf.nn.relu(
                conv2d(normalized2, conv_weights['cw3']) + conv_biases['cb3']
            )
            if num_layers < 5:
                conv3 = normalized2
            else:
                conv3 = max_pool(conv3)
                conv3 = lrn(conv3)

            shape = conv3.get_shape().as_list()
            reshape_dim = [shape[0], shape[1] * shape[2] * shape[3]]
            reshape = tf.reshape(conv3, reshape_dim)
            reshape_cols = reshape_dim[1]
            return reshape, reshape_cols

        train_conv_output, reshape_cols = conv_layers(tf_train_dataset)

        if num_layers == 1:
            hlayer1_nodes == num_labels
        elif num_layers == 2:
            hlayer2_nodes == num_labels
        elif num_layers == 3:
            hlayer3_nodes == num_labels
        else:
            pass

        for i in classifier_matrix():
            with tf.variable_scope("classifier_{}".format(i+1)) as scope:
                weights[i] =  {
                    'h1': tf.get_variable(
                        shape = [reshape_cols, hlayer1_nodes],
                        initializer = tf.contrib.layers.xavier_initializer(),
                        name='h1'
                    ),
                    'h2': tf.get_variable(
                        shape = [hlayer1_nodes, hlayer2_nodes],
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
                out_layer[i] = tf.nn.relu(
                    tf.matmul(data_set, weights[i]['h1']) + biases[i]['b1']
                )
                if training_set(data_set):
                    out_layer[i] = tf.nn.dropout(out_layer[i], keep_prob)
                if num_layers == 1:
                    continue
                out_layer[i] = tf.nn.relu(
                    tf.matmul(out_layer[i], weights[i]['h2']) + biases[i]['b2']
                )
                if training_set(data_set):
                    out_layer[i] = tf.nn.dropout(out_layer[i], keep_prob)
                if num_layers <= 2:
                    continue
                out_layer[i] = tf.nn.relu(
                    tf.matmul(out_layer[i], weights[i]['h3']) + biases[i]['b3']
                )
                if training_set(data_set):
                    out_layer[i] = tf.nn.dropout(out_layer[i], keep_prob)
                if num_layers <= 3:
                    continue
                out_layer[i] = tf.matmul(
                    out_layer[i], weights[i]['out']
                ) + biases[i]['out']
            return out_layer

        def training_set(data_set):
            return data_set == train_conv_output

        def prediction(logits):
            return tf.pack(
                [tf.nn.softmax(logits[i]) for i in classifier_matrix()]
            )

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
        saver = tf.train.Saver()
        print("Initialized")

        if real_life_flag:
            saver.restore(session, "checkpoints/2016-10-31_200k_step_model_7_layers.ckpt")
            print("Model restored")

            real_life_predictions = session.run(
                test_prediction, feed_dict= {
                    tf_test_dataset : real_life_dataset
                }
            )
            predicted_labels = np.argmax(real_life_predictions, 2).T
            label_list = predicted_labels.tolist()

            for i, label in enumerate(label_list):
                joined_label = ''.join(
                    [str(digit) for digit in label[1:5] if digit != 10]
                )
                reshaped_im = real_life_dataset[i][:,:,0]
                plt.imshow(reshaped_im, cmap = 'Greys_r')
                plt.title("Real life image {}".format(i + 1), fontsize = 18)
                plt.text(
                    10,
                    30,
                    'Label: {}'.format(joined_label),
                    fontsize = 22,
                    color = 'r',
                    bbox=dict(facecolor='cyan', alpha=0.7)
                )
                plt.axis('off')
                plt.savefig("real_life_image_{}.png".format(i + 1))
                plt.show(block = False)
                plt.pause(2.0)
                plt.gcf().clear()
                print(label)

        else:
            validation_learning_table = {}
            for step in range(num_steps):
                offset = (
                    step * batch_size) % (train_labels.shape[0] - batch_size
                )
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

                if (step % 5000 == 0):
                    print("Minibatch loss at step %d: %f" % (step, final_loss))
                    minibatch_accuracy = accuracy(predictions, batch_labels)
                    print("Minibatch accuracy: %.1f%%" % minibatch_accuracy)
                    val_accuracy = accuracy(
                        valid_prediction.eval(), valid_labels
                    )
                    print("Validation accuracy: %.1f%%" % val_accuracy)
                    validation_learning_table[step] = val_accuracy

            test_accuracy = accuracy(test_prediction.eval(), test_labels)
            test_result_table.append(test_accuracy)
            print(
                "Test accuracy: %.1f%% with %d layers" % (
                    test_accuracy, num_layers
                )
            )

            working_directory = os.getcwd()
            subfolder = 'checkpoints'
            session_name = "{}_svhn_model_{}_layers.ckpt".format(
                datetime.datetime.now(), num_layers
            )
            filename = os.path.join(
                working_directory,
                subfolder,
                session_name
            )

            save_path = saver.save(session, filename)
            print("Model saved in file: %s" % save_path)

            ordered_table = collections.OrderedDict(
                sorted(validation_learning_table.items())
            )

            zero_buffer = 0.01 * num_steps
            end_buffer = 0.15 * num_steps
            alpha = (0.1 * num_layers + 0.1)

            colors = ['y', 'gray', 'g', 'c', 'purple', 'b', 'red']
            # To indicate division between convolutional and fully-connected
            linestyles = [
                'dashed', 'dashed', 'dashed', 'dashed', 'solid', 'solid', 'solid'
            ]

            plt.ylim([80, 100])
            plt.xlim([0 - zero_buffer, num_steps + end_buffer])
            plt.xlabel('Number of steps')
            plt.ylabel('Digit recognition accuracy (%)')
            plt.plot(
                ordered_table.keys(),
                ordered_table.values(),
                alpha = alpha,
                lw = 3,
                label = "{} layers".format(num_layers),
                color = colors[num_layers - 1],
                ls = linestyles[num_layers - 1]
            )
            plt.scatter(
                num_steps,
                test_accuracy,
                alpha = alpha,
                s = num_layers * 50,
                color = colors[num_layers - 1]
            )

            lap_time_end = timeit.default_timer()
            lap_time = lap_time_end - lap_time_start
            lap_time_in_minutes = round(lap_time / 60)
            print(
                'Recognition session with {} layers lasted {} minutes'.format(
                    num_layers, lap_time_in_minutes
                )
            )

            finish_time = timeit.default_timer()
            process_time = finish_time - start_time
            process_time_in_minutes = round(process_time / 60)

            print('Total processing time was {} minutes'.format(process_time_in_minutes))
            plt.legend(loc = 4)

            plt.annotate(
                'Validation accuracy',
                xy=(num_steps / 2, 95),
                xytext=(num_steps / 4, 98),
                arrowprops=dict(facecolor='black', shrink=0.05),
            )
            plt.annotate(
                'Test accuracy',
                xy=(num_steps, 96),
                xytext=(3 * num_steps / 4, 98),
                arrowprops=dict(facecolor='black', shrink=0.05),
            )
            for acc in test_result_table:
                test_result = str(round(acc, 1)) + '%'
                plt.annotate(test_result, xy=(num_steps * 1.04, acc))

            plt.suptitle('Digit Sequence Learning Performance', size=12, y=0.97)
            plt.title('For various neural networks layer numbers', fontsize=8, y=1.0)
            plt.savefig('learning_rate_chart.png')
            plt.show()



