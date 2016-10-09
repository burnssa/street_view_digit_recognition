### Deep learning capstone project
### Extracting data for training and testing

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

# %matplotlib inline

# Note - code taken from Assignment 1 of Udacity Deep Learning course

class ExtractNotMNISTImageData():

    def __init__(self):
    self.url = 'http://commondatastorage.googleapis.com/books1000/'
    self.last_percent_reported = None
    self.num_classes = 10
    # np.random.seed(133)

    def download_progress_hook(count, blockSize, totalSize):
        """A hook to report the progress of a download. This is mostly intended for users with
        slow internet connections. Reports every 1% change in download progress.
        """
        global last_percent_reported
        percent = int(count * blockSize * 100 / totalSize)

        if last_percent_reported != percent:
            if percent % 5 == 0:
                sys.stdout.write("%s%%" % percent)
                sys.stdout.flush()
            else:
                sys.stdout.write(".")
                sys.stdout.flush()

            last_percent_reported = percent

    def maybe_download(self, filename, expected_bytes, force=False):
        """Download a file if not present, and make sure it's the right size."""
        if force or not os.path.exists(filename):
            print('Attempting to download:', filename)
            filename, _ = urlretrieve(self.url + filename, filename, reporthook=download_progress_hook)
            print('\nDownload Complete!')
        statinfo = os.stat(filename)
        if statinfo.st_size == expected_bytes:
            print('Found and verified', filename)
        else:
          raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
        return filename

    def maybe_extract(self, filename, force=False):
        root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
        if os.path.isdir(root) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping extraction of %s.' % (root, filename))
        else:
            print('Extracting data for %s. This may take a while. Please wait.' % root)
            tar = tarfile.open(filename)
            sys.stdout.flush()
            tar.extractall()
            tar.close()
        data_folders = [
            os.path.join(root, d) for d in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, d))]
        if len(data_folders) != self.num_classes:
            raise Exception(
                'Expected %d folders, one per class. Found %d instead.' % (
                self.num_classes, len(data_folders)))
        print(data_folders)
        return data_folders


