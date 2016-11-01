import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import pdb
from six.moves import cPickle as pickle
import h5py

from PIL import Image, ImageDraw

train_file = os.path.join('train', 'digitStruct.mat')
h5py_train = h5py.File(train_file, 'r')

extra_file = os.path.join('train', 'digitStruct.mat')
h5py_extra = h5py.File(extra_file, 'r')

def example_digit_struct(h5py, index):
    digit_struct_dict = {}
    digit_struct_dict['name'] = h5py['digitStruct']['name'][index]
    digit_struct_dict['bbox'] = h5py['digitStruct']['bbox'][index]
    return digit_struct_dict

def find_four_digit_number_index():
    digits = 1
    while digits != 4:
        digit_struct_count = h5py_train['digitStruct']['name'].shape[0]
        index = int(random.random() * digit_struct_count)
        digit_struct = example_digit_struct(h5py_train, index)
        digits = h5py_train[digit_struct['bbox'][0]]['top'].shape[0]
    return index

index = find_four_digit_number_index()
digit_struct = example_digit_struct(h5py_train, index)

filename_characters = h5py_train[digit_struct['name'][0]].value
filename_string = ''.join([chr(c) for c in filename_characters])
full_filename = 'train/' + filename_string

def get_bbox(dataset, bbox_array):
    bbox = {}
    dimensions = dataset[bbox_array[0]].keys()
    for dim in dimensions:
        bbox[dim] = dataset[bbox_array[0]][dim]
    return bbox

bbox = get_bbox(h5py_train, digit_struct['bbox'])

def draw_box(plt, top, left, width, height):
    plt.plot([left, left + width], [top, top], color='k', ls='-', lw=2)
    plt.plot([left, left], [top + height, top], color='k', ls='-', lw=2)
    plt.plot(
        [left, left + width],
        [top + height, top + height],
        color='k',
        ls='-',
        lw=2
    )
    plt.plot(
        [left + width, left + width],
        [top + height, top],
        color='k',
        ls='-',
        lw=2
    )

def dimension_helper(dataset, dimension, index):
    if dimension.shape[0] > 1:
        box_value = dataset[dimension.value[index].item()].value[0][0]
    else:
        box_value = dimension.value[index].item()
    return box_value

def display_example(dataset, bbox, full_filename):
    im = mpimg.imread(full_filename)
    displayed_digits = len(bbox[bbox.keys()[0]])

    for i in range(displayed_digits):
        draw_box(
            plt,
            dimension_helper(dataset, bbox['top'], i),
            dimension_helper(dataset, bbox['left'], i),
            dimension_helper(dataset, bbox['width'], i),
            dimension_helper(dataset, bbox['height'], i)
        )
    plt.imshow(im)
    plt.savefig('example_digit_image.png')
    plt.show()

display_example(h5py_train, bbox, full_filename)

def get_summary_dimension_statistics(h5py_dataset):
    width_list = []
    height_list = []
    bbox_set = h5py_dataset['digitStruct']['bbox']

    for row in bbox_set:
        bbox = get_bbox(h5py_dataset, row)
        digits = len(bbox[bbox.keys()[0]])
        digit_widths = []
        digit_heights = []

        for i in range(digits):
            digit_widths.append(
                dimension_helper(h5py_dataset, bbox['width'], i)
            )
            digit_heights.append(
                dimension_helper(h5py_dataset, bbox['height'], i)
            )
        max_width = max(digit_widths)
        max_height = max(digit_heights)

        width_list.append(max_width)
        height_list.append(max_height)

        width_array = np.asarray(width_list)
        height_array = np.asarray(height_list)
    return width_array, height_array

train_bbox_set = h5py_train['digitStruct']['bbox']
extra_bbox_set = h5py_extra['digitStruct']['bbox']

print('Getting summary statistics')

train_widths, train_heights = get_summary_dimension_statistics(h5py_train)
extra_widths, extra_heights = get_summary_dimension_statistics(h5py_extra)

widths = np.append(train_widths, extra_widths)
heights = np.append(train_heights, extra_heights)

def print_summary_dimension_data(widths, heights):
    max_width_percentile = np.percentile(widths, 98)
    max_height_percentile = np.percentile(heights, 98)

    plt.hist(widths, bins='auto')
    plt.title('Histogram of training data digit box widths')
    plt.xlabel('Pixel size bins')
    plt.xlim(0, max_width_percentile)
    plt.savefig('digit_box_width_histogram.png')
    plt.show()

    plt.hist(heights, bins='auto')
    plt.title('Histogram of training data digit box heights')
    plt.xlabel('Pixel size bins')
    plt.xlim(0, max_height_percentile)
    plt.savefig('digit_box_height_histogram.png')
    plt.show()

    print('Average digit box width: {}'.format(np.mean(widths)))
    print('Max digit box width: {}'.format(np.max(widths)))
    print('98th percentile digit box width: {}'.format(max_width_percentile))

    print('Average digit box height: {}'.format(np.mean(heights)))
    print('Max digit box height: {}'.format(np.max(heights)))
    print('98th percentile digit box height: {}'.format(max_height_percentile))

print('Printing histogram')

print_summary_dimension_data(widths, heights)
