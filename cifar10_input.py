# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cPickle
import re
import random
import numpy as np
from PIL import Image

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_nn_ops

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 32

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def unpickle(relpath):
    with open(relpath, 'rb') as fp:
        d = cPickle.load(fp)
    return d


def prepare_input(data=None, labels=None):
    assert data.shape[1] == IMAGE_SIZE * IMAGE_SIZE * 3
    assert data.shape[0] == labels.shape[0]
    # data is transformed from (no_of_samples, 3072)
    # to (no_of_samples , image_height, image_width, image_depth)
    # make sure the type of the data is no.float32
    data = data.reshape([-1, 3, IMAGE_SIZE, IMAGE_SIZE])
    data = data.transpose([0, 2, 3, 1])
    data = data.astype(np.float32)

    image_mean_by_channel = [125.3, 123.0, 113.9]
    stddev = [63.0, 62.1, 66.7]
    # color nomalization
    data = data - image_mean_by_channel
    data = data / stddev

    is_nan = np.isnan(data)
    is_inf = np.isinf(data)
    if np.any(is_nan) or np.any(is_inf):
        print('data is not well-formed : is_nan {n}, is_inf: {i}'.format(
            n=np.any(is_nan), i=np.any(is_inf)))

    # TODO: cant add into tensorboard QQ
    # image = tf.constant(data[0].astype(np.float32), name='image')
    # tf.summary.image('eval_input_image', image)

    return data, labels


def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                                    nw:nw + crop_shape[1]]
    return np.array(new_batch)


def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch


def prepare_input_with_augmentation(data=None, labels=None):
    assert data.shape[1] == IMAGE_SIZE * IMAGE_SIZE * 3
    assert data.shape[0] == labels.shape[0]
    # data is transformed from (no_of_samples, 3072)
    # to (no_of_samples , image_height, image_width, image_depth)
    # make sure the type of the data is no.float32
    data = data.reshape([-1, 3, IMAGE_SIZE, IMAGE_SIZE])
    data = data.transpose([0, 2, 3, 1])

    # color nomalization
    image_mean_by_channel = [125.3, 123.0, 113.9]
    stddev = [63.0, 62.1, 66.7]
    data = data - image_mean_by_channel
    data = data / stddev

    is_nan = np.isnan(data)
    is_inf = np.isinf(data)

    if np.any(is_nan) or np.any(is_inf):
        print('data is not well-formed : is_nan {n}, is_inf: {i}'.format(
            n=np.any(is_nan), i=np.any(is_inf)))

    # padding
    padnCrop_data = _random_crop(data, [32, 32], 4)
    # flip left right
    data = _random_flip_leftright(padnCrop_data)

    data = data.astype(np.float32)

    return data, labels


def load_and_preprocess_input(dataset_dir=None):
    assert os.path.isdir(dataset_dir)
    train_all = {'data': [], 'labels': []}
    trn_all_data = []
    trn_all_labels = []
    validate_all = {'data': [], 'labels': []}
    vldte_all_data = []
    vldte_all_labels = []

    r_data_file = re.compile('^data_batch_\\d+')
    # for loading train dataset, iterate through the directory to get matchig
    # data file
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            m = r_data_file.match(f)
            if m:
                relpath = os.path.join(root, f)
                d = unpickle(os.path.join(root, f))
                trn_all_data.append(d['data'])
                trn_all_labels.append(d['labels'])
    # concatenate all the  data in various files into one ndarray of shape
    # data.shape == (no_of_samples, 3072), where 3072=image_depth x image_height x image_width
    #labels.shape== (no_of_samples)
    trn_all_data, trn_all_labels = (np.concatenate(trn_all_data).astype(np.float32),
                                    np.concatenate(trn_all_labels).astype(np.int32))

    # load the only test data set for validation and testing
    # use only the first n_validate_samples samples for validating
    test_temp = unpickle(os.path.join(dataset_dir, 'test_batch'))
    vldte_all_data = test_temp['data'][0:(NUM_EXAMPLES_PER_EPOCH_FOR_EVAL), :]
    vldte_all_labels = test_temp['labels'][0:(NUM_EXAMPLES_PER_EPOCH_FOR_EVAL)]
    vldte_all_data, vldte_all_labels = (np.concatenate([vldte_all_data]).astype(np.float32),
                                        np.concatenate([vldte_all_labels]).astype(np.int32))
    # transform the test images in the same manner as the train images
    train_all['data'], train_all['labels'] = prepare_input_with_augmentation(
        data=trn_all_data, labels=trn_all_labels)
    validate_and_test_data, validate_and_test_labels = prepare_input(
        data=vldte_all_data, labels=vldte_all_labels)

    validate_all['data'] = validate_and_test_data[0:NUM_EXAMPLES_PER_EPOCH_FOR_EVAL, :, :, :]
    validate_all['labels'] = validate_and_test_labels[0:NUM_EXAMPLES_PER_EPOCH_FOR_EVAL]

    # load all label-names
    print('train_all: ', 'data: ', train_all['data'].shape, train_all['data'].dtype,
          '  labels: ', train_all['labels'].shape, train_all['labels'].dtype)
    label_names_for_validation_and_test = unpickle(
        os.path.join(dataset_dir, 'batches.meta'))['label_names']
    print ('validate_all: ', 'data: ', validate_all['data'].shape, validate_all['data'].dtype,
           '  labels: ', validate_all['labels'].shape, validate_all['labels'].dtype)
    print ('label_names_for_validation_and_test',
           label_names_for_validation_and_test)
    return train_all, validate_all

