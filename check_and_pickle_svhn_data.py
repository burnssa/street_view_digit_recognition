### Deep learning capstone project
### Loading SVHN data for classification

## Draws on work by https://goo.gl/0IgPJZ
## Also references https://goo.gl/szj3l1

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
from scipy import ndimage
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import random
import h5py
import PIL.Image as Image
import matplotlib.image as mpimg
from scipy.misc import imresize

url = 'http://ufldl.stanford.edu/housenumbers/'
last_percent_reported = None

def download_progress_hook(count, blockSize, totalSize):
    """
    Report every 1% change in download progress.
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

def maybe_download(filename, force=False):
    """
    Download a file if not present, and make sure it's the right size.
    """
    if force or not os.path.exists(filename):
        print('Attempting to download:', filename)
        filename, _ = urlretrieve(
            url + filename, filename, reporthook=download_progress_hook
        )
        print('\nDownload Complete!')
        statinfo = os.stat(filename)
    return filename

train_filename = maybe_download('train.tar.gz')
test_filename = maybe_download('test.tar.gz')
extra_filename = maybe_download('extra.tar.gz')

np.random.seed(133)

def maybe_extract(filename, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        print(
            '%s already present - Skipping extraction of %s.' % (root, filename)
        )
    else:
        print(
            'Extracting data for %s. This may take a while. Please wait.' % root
        )
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall()
        tar.close()
    data_folders = root
    print(data_folders)
    return data_folders

train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)
extra_folders = maybe_extract(extra_filename)

class DigitStructFile:
    def __init__(self, inf):
        self.inf = h5py.File(inf, 'r')
        self.digitStructName = self.inf['digitStruct']['name']
        self.digitStructBbox = self.inf['digitStruct']['bbox']

    def getName(self,n):
        return ''.join(
            [chr(c[0]) for c in self.inf[self.digitStructName[n][0]].value]
        )

    def bboxHelper(self,attr):
        if (len(attr) > 1):
            attr = [
                self
                .inf[attr.value[j]
                .item()]
                .value[0][0] for j in range(len(attr))
            ]
        else:
            attr = [attr.value[0][0]]
        return attr

    def getBbox(self,n):
        bbox = {}
        bb = self.digitStructBbox[n].item()
        bbox['height'] = self.bboxHelper(self.inf[bb]["height"])
        bbox['label'] = self.bboxHelper(self.inf[bb]["label"])
        bbox['left'] = self.bboxHelper(self.inf[bb]["left"])
        bbox['top'] = self.bboxHelper(self.inf[bb]["top"])
        bbox['width'] = self.bboxHelper(self.inf[bb]["width"])
        return bbox

    def getDigitStructure(self,n):
        s = self.getBbox(n)
        s['name']=self.getName(n)
        return s

    def getAllDigitStructure(self):
        return [
            self.getDigitStructure(i) for i in range(len(self.digitStructName))
        ]

    def getAllDigitStructureByDigit(self):
        pictDat = self.getAllDigitStructure()
        result = []
        structCnt = 1
        for i in range(len(pictDat)):
            item = { 'filename' : pictDat[i]["name"] }
            figures = []
            for j in range(len(pictDat[i]['height'])):
                figure = {}
                figure['height'] = pictDat[i]['height'][j]
                figure['label']  = pictDat[i]['label'][j]
                figure['left']   = pictDat[i]['left'][j]
                figure['top']    = pictDat[i]['top'][j]
                figure['width']  = pictDat[i]['width'][j]
                figures.append(figure)
            structCnt = structCnt + 1
            item['boxes'] = figures
            result.append(item)
        return result

train_folders = 'train'
test_folders = 'test'
extra_folders = 'extra'

fin = os.path.join(train_folders, 'digitStruct.mat')
dsf = DigitStructFile(fin)
train_data = dsf.getAllDigitStructureByDigit()

fin = os.path.join(test_folders, 'digitStruct.mat')
dsf = DigitStructFile(fin)
test_data = dsf.getAllDigitStructureByDigit()

fin = os.path.join(extra_folders, 'digitStruct.mat')
dsf = DigitStructFile(fin)
extra_data = dsf.getAllDigitStructureByDigit()

# # Guidance on converting RGB image to grayscale
# # http://stackoverflow.com/questions/687261/converting-rgb-to-grayscale-intensity
rgb_to_grayscale_factors = [[0.2989],[0.5870],[0.1140]]

def generate_dataset(data, folder):
    dataset = np.ndarray([len(data),32,32,1], dtype='float32')
    labels = np.ones([len(data),6], dtype=int) * 10
    for i in np.arange(len(data)):
        filename = data[i]['filename']
        fullname = os.path.join(folder, filename)
        im = Image.open(fullname)
        boxes = data[i]['boxes']
        num_digit = len(boxes)
        labels[i,0] = num_digit
        top = np.ndarray([num_digit], dtype='float32')
        left = np.ndarray([num_digit], dtype='float32')
        height = np.ndarray([num_digit], dtype='float32')
        width = np.ndarray([num_digit], dtype='float32')
        for j in np.arange(num_digit):
            if j < 5:
                labels[i,j+1] = boxes[j]['label']
                if boxes[j]['label'] == 10: labels[i,j+1] = 0
            else: print('#',i,'image has more than 5 digits.')
            top[j] = boxes[j]['top']
            left[j] = boxes[j]['left']
            height[j] = boxes[j]['height']
            width[j] = boxes[j]['width']

        im_top = np.amin(top)
        im_left = np.amin(left)
        im_height = np.amax(top) + height[np.argmax(top)] - im_top
        im_width = np.amax(left) + width[np.argmax(left)] - im_left

        im_top = np.floor(im_top - 0.1 * im_height)
        im_left = np.floor(im_left - 0.1 * im_width)
        im_bottom = np.amin([np.ceil(im_top + 1.2 * im_height), im.size[1]])
        im_right = np.amin([np.ceil(im_left + 1.2 * im_width), im.size[0]])

        im = im.crop(
            (im_left, im_top, im_right, im_bottom)
        ).resize([32,32], Image.ANTIALIAS)
        im = np.dot(np.array(im, dtype='float32'), rgb_to_grayscale_factors)
        mean = np.mean(im, dtype='float32')
        std = np.std(im, dtype='float32', ddof=1)
        if std < 1e-4: std = 1.
        im = (im - mean) / std
        dataset[i,:,:,:] = im[:,:,:]

    return dataset, labels

train_dataset, train_labels = generate_dataset(train_data, train_folders)
print(train_dataset.shape, train_labels.shape)

test_dataset, test_labels = generate_dataset(test_data, test_folders)
print(test_dataset.shape, test_labels.shape)

extra_dataset, extra_labels = generate_dataset(extra_data, extra_folders)
print(extra_dataset.shape, extra_labels.shape)

random.seed()

n_labels = 10
valid_index = []
valid_index2 = []
train_index = []
train_index2 = []
for i in np.arange(n_labels):
    valid_index.extend(np.where(train_labels[:,1] == (i))[0][:400].tolist())
    train_index.extend(np.where(train_labels[:,1] == (i))[0][400:].tolist())
    valid_index2.extend(np.where(extra_labels[:,1] == (i))[0][:200].tolist())
    train_index2.extend(np.where(extra_labels[:,1] == (i))[0][200:].tolist())

random.shuffle(valid_index)
random.shuffle(train_index)
random.shuffle(valid_index2)
random.shuffle(train_index2)

valid_dataset = np.concatenate(
    (extra_dataset[valid_index2,:,:,:], train_dataset[valid_index,:,:,:]),
    axis=0
)
valid_labels = np.concatenate(
    (extra_labels[valid_index2,:], train_labels[valid_index,:]),
    axis=0
)
train_dataset_t = np.concatenate(
    (extra_dataset[train_index2,:,:,:], train_dataset[train_index,:,:,:]),
    axis=0
)
train_labels_t = np.concatenate(
    (extra_labels[train_index2,:], train_labels[train_index,:]),
    axis=0
)

dataset_foldername = 'real_life_digit_samples'

def resize(im):
    resized = imresize(im, (32, 32))
    return resized

def grayscale(im):
    gray = np.dot(np.array(im, dtype='float32'), rgb_to_grayscale_factors)
    return gray

def process_live_data(data_folder):
    image_dataset = []
    folders_and_files =  os.listdir('real_life_digit_samples')
    image_files = [
        image_file for image_file in folders_and_files if
            image_file.endswith('.png')
    ]

    for i, image_file in enumerate(image_files):
        full_filename = dataset_foldername + '/' + image_file
        im = mpimg.imread(full_filename)
        plt.imshow(im)
        plt.show(block = False)
        plt.pause(0.001)

        # Converting from rgba to rgb
        rgb_im = im[:,:,:3]
        resized_im = resize(rgb_im)
        gray_im = grayscale(resized_im)

        mean = np.mean(gray_im, dtype='float32')
        std = np.std(gray_im, dtype='float32', ddof=1)
        if std < 1e-4: std = 1.
        normalized_im = (gray_im - mean) / std
        image_dataset.append(normalized_im)
        image_data_array = np.asarray(image_dataset)

    return image_data_array

live_dataset = process_live_data(dataset_foldername)

print(train_dataset_t.shape, train_labels_t.shape)
print(test_dataset.shape, test_labels.shape)
print(valid_dataset.shape, valid_labels.shape)
print(valid_dataset.shape, valid_labels.shape)

print(live_dataset.shape)

pickle_file = 'SVHN_multi.pickle'

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_dataset_t,
        'train_labels': train_labels_t,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
        'real_life_dataset': live_dataset
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)
