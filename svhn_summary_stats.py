import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import pdb
from six.moves import cPickle as pickle

# from check_and_pickle_svhn_data import digitStructFile
import h5py

from PIL import Image, ImageDraw

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

def get_distribution_of_digits(data_labels):
    number_digits = data_labels[:,0]
    unique, counts = np.unique(number_digits, return_counts=True)
    distribution_dict = dict(zip(unique, counts))
    for unique, counts in distribution_dict.iteritems():
        print('Digit case: {} | Occurrences: {}'.format(unique, counts))

def get_average_number_of_digits(data_labels):
    number_digits = data_labels[:,0]
    avg_number_digits = np.mean(number_digits)
    print('Average number of digits: {}'.format(avg_number_digits))

def get_dataset_rows(dataset):
    return dataset.shape[0]

six_digit_index = np.where(train_labels[:,0] == 6)[0][0]
train_labels = np.delete(train_labels, six_digit_index, 0)
train_dataset = np.delete(train_dataset, six_digit_index, 0)

print('Train Dataset - summary stats')
print('Dataset rows: {}'.format(get_dataset_rows(train_dataset)))
get_average_number_of_digits(train_labels)
get_distribution_of_digits(train_labels)

print('\n')
print('Validation Dataset - summary stats')
print('Dataset rows: {}'.format(get_dataset_rows(valid_dataset)))
get_average_number_of_digits(valid_labels)
get_distribution_of_digits(valid_labels)

print('\n')
print('Test Dataset - summary stats')
print('Dataset rows: {}'.format(get_dataset_rows(test_dataset)))
get_average_number_of_digits(test_labels)
get_distribution_of_digits(test_labels)





