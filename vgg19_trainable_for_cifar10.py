from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import numpy as np
from functools import reduce

VGG_MEAN = [103.939, 116.779, 123.68]  # BGR


class Vgg19:
    """
    A trainable version VGG19.
    """

    def __init__(self, vgg19_npy_path=None, trainable=True, dropout=0.5, wd=None):
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.dropout = dropout
        self.wd = wd

    def loss(self, logits, labels, train_mode):
        if train_mode is True:
            # Calculate the average cross entropy loss across the batch.
            labels = tf.cast(labels, tf.int64)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits, name='cross_entropy_per_example')
            cross_entropy_mean = tf.reduce_mean(
                cross_entropy, name='cross_entropy')
            tf.add_to_collection('losses', cross_entropy_mean)

            # # The total loss is defined as the cross entropy loss plus all of the weight
            # # decay terms (L2 loss).
            return tf.add_n(tf.get_collection('losses'), name='total_loss')
        else:
            # Calculate the average cross entropy loss across the batch.
            labels = tf.cast(labels, tf.int64)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits, name='cross_entropy_per_example_test')
            cross_entropy_mean = tf.reduce_mean(
                cross_entropy, name='cross_entropy_test')

            return cross_entropy_mean

    def build(self, rgb, train_mode=None):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """
        self.conv1_1 = self.conv_layer(rgb, 3, 64, "conv1_1", train_mode)
        self.conv1_2 = self.conv_layer(
            self.conv1_1, 64, 64, "conv1_2", train_mode)
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')  # 16*16

        self.pool1 = tf.contrib.layers.batch_norm(inputs=self.pool1)

        self.conv2_1 = self.conv_layer(
            self.pool1, 64, 128, "conv2_1", train_mode)
        self.conv2_2 = self.conv_layer(
            self.conv2_1, 128, 128, "conv2_2", train_mode)
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')  # 8*8

        self.pool2 = tf.contrib.layers.batch_norm(inputs=self.pool2)

        self.conv3_1 = self.conv_layer(
            self.pool2, 128, 256, "conv3_1", train_mode)
        self.conv3_2 = self.conv_layer(
            self.conv3_1, 256, 256, "conv3_2", train_mode)
        self.conv3_3 = self.conv_layer(
            self.conv3_2, 256, 256, "conv3_3", train_mode)
        self.conv3_4 = self.conv_layer(
            self.conv3_3, 256, 256, "conv3_4", train_mode)
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')  # 4*4

        self.pool3 = tf.contrib.layers.batch_norm(inputs=self.pool3)

        self.conv4_1 = self.conv_layer(
            self.pool3, 256, 512, "conv4_1", train_mode)
        self.conv4_2 = self.conv_layer(
            self.conv4_1, 512, 512, "conv4_2", train_mode)
        self.conv4_3 = self.conv_layer(
            self.conv4_2, 512, 512, "conv4_3", train_mode)
        self.conv4_4 = self.conv_layer(
            self.conv4_3, 512, 512, "conv4_4", train_mode)
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')  # 2*2

        self.pool4 = tf.contrib.layers.batch_norm(inputs=self.pool4)

        self.conv5_1 = self.conv_layer(
            self.pool4, 512, 512, "conv5_1", train_mode)
        self.conv5_2 = self.conv_layer(
            self.conv5_1, 512, 512, "conv5_2", train_mode)
        self.conv5_3 = self.conv_layer(
            self.conv5_2, 512, 512, "conv5_3", train_mode)
        self.conv5_4 = self.conv_layer(
            self.conv5_3, 512, 512, "conv5_4", train_mode)

        self.conv5_4 = tf.contrib.layers.batch_norm(inputs=self.conv5_4)

        self.flat5 = tf.contrib.layers.flatten(self.conv5_4, "flat5")
        # 2048 = 2*2*512
        # self.fc6 = self.fc_layer(self.flat5, 2048, 4096, "fc6")
        self.fc6 = self.fc_layer(
            self.flat5, 2048, 4096, "fc6_random_init", train_mode)
        self.relu6 = tf.nn.relu(self.fc6)

        # dropout
        if train_mode is not None:
            self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(
                self.relu6, self.dropout), lambda: self.relu6)
        elif self.trainable:
            self.relu6 = tf.nn.dropout(self.relu6, self.dropout)

        self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7", train_mode)
        self.relu7 = tf.nn.relu(self.fc7)

        # dropout
        if train_mode is not None:
            self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(
                self.relu7, self.dropout), lambda: self.relu7)
        elif self.trainable:
            self.relu7 = tf.nn.dropout(self.relu7, self.dropout)

        # self.fc8 = self.fc_layer(self.relu7, 4096, 10, "fc8")
        self.fc8 = self.fc_layer(self.relu7, 4096, 10,
                                 "fc8_random_init", train_mode)

        self.logits = self.fc8

        self.data_dict = None

        return self.logits

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name, train_mode):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(
                3, in_channels, out_channels, name, train_mode)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name, train_mode):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(
                in_size, out_size, name, train_mode)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name, train_mode):
        initial_value = tf.truncated_normal_initializer(
            mean=0.0, stddev=0.03, dtype=tf.float32)
        filters = self.get_var(initial_value, [
                               filter_size, filter_size, in_channels, out_channels], name, 0, name + "_filters", train_mode)

        initial_value = tf.random_normal_initializer(
            mean=0.0, stddev=0.001, dtype=tf.float32)
        biases = self.get_bias_var(
            initial_value, [out_channels], name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name, train_mode):
        initial_value = tf.truncated_normal_initializer(
            mean=0.0, stddev=0.01, dtype=tf.float32)
        weights = self.get_var(
            initial_value, [in_size, out_size], name, 0, name + "_weights", train_mode)

        initial_value = tf.random_normal_initializer(
            mean=0.0, stddev=0.001, dtype=tf.float32)
        biases = self.get_bias_var(
            initial_value, [out_size], name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, shape, name, idx, var_name, train_mode):

        # value = initial_value
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
            with tf.device('/cpu:0'):
                if self.trainable:
                    # If initializer is a constant, do not specify shape.
                    var = tf.get_variable(
                        name=var_name, initializer=value, dtype=tf.float32)
                else:
                    var = tf.constant(value, dtype=tf.float32, name=var_name)
        else:
            value = initial_value
            with tf.device('/cpu:0'):
                if self.trainable:
                    var = tf.get_variable(
                        name=var_name, shape=shape, initializer=value, dtype=tf.float32)
                else:
                    var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        print (var_name, var.get_shape())

        if self.wd is not None:
            if train_mode is not None:
                if train_mode is True:
                    weight_decay = tf.multiply(
                        tf.nn.l2_loss(var), self.wd, name='weight_loss')
                    tf.add_to_collection('losses', weight_decay)
            else:
                pass

        return var

    def get_bias_var(self, initial_value, shape, name, idx, var_name):

        # value = initial_value
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
            with tf.device('/cpu:0'):
                if self.trainable:
                    # If initializer is a constant, do not specify shape.
                    var = tf.get_variable(
                        name=var_name, initializer=value, dtype=tf.float32)
                else:
                    var = tf.constant(value, dtype=tf.float32, name=var_name)
        else:
            value = initial_value
            with tf.device('/cpu:0'):
                if self.trainable:
                    var = tf.get_variable(
                        name=var_name, shape=shape, initializer=value, dtype=tf.float32)
                else:
                    var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        print (var_name, var.get_shape())

        return var

    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count
