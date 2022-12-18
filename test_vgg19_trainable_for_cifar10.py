"""
Simple tester for the vgg19_trainable
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile
import math
import time
from datetime import datetime
import tensorflow as tf
import numpy as np
from six.moves import urllib
import cifar10_input
import vgg19_trainable_for_cifar10 as vgg19

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 391 * 164,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
DATA_URL_PYTHON = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
MOMENTUM_RATE = 0.9


def train(train_all, validate_all):
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default() as g:
        global_step = tf.contrib.framework.get_or_create_global_step()

        # training
        n_batches_train = int(math.floor(
            train_all['labels'].shape[0] / FLAGS.batch_size))
        # validation
        n_batches_validate = int(
            math.floor(validate_all['labels'].shape[0] / FLAGS.batch_size))

        imagesss = tf.placeholder(
            tf.float32, shape=(FLAGS.batch_size, 32, 32, 3))
        labelsss = tf.placeholder(tf.int32, shape=(FLAGS.batch_size))
        train_mode = tf.placeholder(tf.bool)

        with tf.variable_scope('lab3'):
            vgg = vgg19.Vgg19('./pre_trained/vgg19.npy',
                              trainable=True, dropout=0.5, wd=None)
            train_logits = vgg.build(imagesss, train_mode)
            tf.get_variable_scope().reuse_variables()
            test_logits = vgg.build(imagesss, train_mode)

        train_loss = vgg.loss(train_logits, labelsss, train_mode=train_mode)
        top_k_train = tf.nn.in_top_k(train_logits, labelsss, 1)
        num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
        learning_rate = tf.train.piecewise_constant(global_step,
                                                    [81 * tf.to_int64(num_batches_per_epoch),
                                                     122 * tf.to_int64(num_batches_per_epoch)],
                                                    [0.0100,
                                                     0.0010,
                                                     0.0001])
        train_op = tf.train.MomentumOptimizer(
            learning_rate, MOMENTUM_RATE, use_nesterov=True).minimize(train_loss)

        # print number of variables used: 45239370 variables, i.e. ideal size =
        # MB
        print (vgg.get_var_count())

        test_loss = vgg.loss(test_logits, labelsss, train_mode=train_mode)
        top_k_test = tf.nn.in_top_k(test_logits, labelsss, 1)

        init = tf.global_variables_initializer()

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Add summary
        # summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=g)
        summary = tf.Summary()

        def all_batches_run_train(n_batches, data=None, labels=None):
            true_count = 0
            loss_count = 0
            sum_n_samples = 0
            for b in xrange(n_batches):
                offset = b * FLAGS.batch_size
                batch_data = data[offset: offset +
                                  FLAGS.batch_size, :, :, :]
                n_samples = batch_data.shape[0]
                batch_labels = labels[offset: offset + FLAGS.batch_size]
                feed_dict = {imagesss: batch_data,
                             labelsss: batch_labels,
                             train_mode: True}
                _, loss_value, top_k_result = sess.run([train_op, train_loss, top_k_train],
                                                       feed_dict=feed_dict)
                true_count += np.sum(top_k_result)
                loss_count += loss_value * n_samples
                sum_n_samples += n_samples
                if n_samples != FLAGS.batch_size:
                    print ('n_samples =%d' % n_samples)
            return loss_count / sum_n_samples, true_count / sum_n_samples

        def all_batches_run_test(n_batches, data=None, labels=None):
            true_count = 0
            loss_count = 0
            sum_n_samples = 0
            for b in xrange(n_batches):
                offset = b * FLAGS.batch_size
                batch_data = data[offset: offset +
                                  FLAGS.batch_size, :, :, :]
                n_samples = batch_data.shape[0]
                batch_labels = labels[offset: offset + FLAGS.batch_size]
                feed_dict = {imagesss: batch_data,
                             labelsss: batch_labels,
                             train_mode: False}
                loss_value, top_k_result = sess.run([test_loss, top_k_test],
                                                    feed_dict=feed_dict)
                true_count += np.sum(top_k_result)
                loss_count += loss_value * n_samples
                sum_n_samples += n_samples
                if n_samples != FLAGS.batch_size:
                    print ('n_samples =%d' % n_samples)
            return loss_count / sum_n_samples, true_count / sum_n_samples

        with tf.Session() as sess:
            sess.run(init)
            for step in xrange(FLAGS.max_steps):
                if step % 391 == 0 or (step + 1) == FLAGS.max_steps:
                    start_time = time.time()
                    n_data = train_all['data'].shape[0]
                    perm = np.random.permutation(n_data)
                    permuted_data = train_all['data'][perm, :, :, :]
                    permuted_labels = train_all['labels'][perm]

                    mean_loss_per_sample_train, precision_per_sample_train = all_batches_run_train(
                        n_batches_train, data=permuted_data, labels=permuted_labels)

                    assert not np.isnan(
                        mean_loss_per_sample_train), 'Model diverged with loss = NaN'

                    duration = time.time() - start_time
                    num_examples_per_epoch = FLAGS.batch_size * 391
                    examples_per_sec = num_examples_per_epoch / duration
                    sec_per_epoch = float(duration)

                    validate_data = validate_all['data']
                    validate_labels = validate_all['labels']

                    mean_loss_per_sample_test, precision_per_sample_test = all_batches_run_test(
                        n_batches_validate, data=validate_data, labels=validate_labels)

                    format_str = ('%s: step %d, loss_train = %.3f, prec_train = %.3f (%.1f examples/sec; %.3f '
                                  'sec/epoch), prec_test = %.3f, loss_test = %.3f')
                    print (format_str % (datetime.now(), step, mean_loss_per_sample_train, precision_per_sample_train,
                                         examples_per_sec, sec_per_epoch, precision_per_sample_test,
                                         mean_loss_per_sample_test))

                    # summary.ParseFromString(sess.run(summary_op))
                    summary.value.add(tag='Training Accuracy',
                                      simple_value=precision_per_sample_train)
                    summary.value.add(tag='Training Loss',
                                      simple_value=mean_loss_per_sample_train)
                    summary.value.add(tag='Testing Accuracy',
                                      simple_value=precision_per_sample_test)
                    summary.value.add(tag='Testing Loss',
                                      simple_value=mean_loss_per_sample_test)
                    summary_writer.add_summary(summary, step)

                    # Save the model checkpoint periodically.
                    if (step + 1) % 3910 == 0 or (step + 1) == FLAGS.max_steps:
                        model_name = 'model.ckpt'
                        checkpoint_path = os.path.join(
                            FLAGS.train_dir, model_name)
                        saver.save(sess, checkpoint_path, global_step=step)

        # vgg.save_npy(sess, './saved_model/test-save.npy')


def maybe_download_and_extract(URL):
    """Download and extract the tarball from Alex's website."""
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%'
                             % (filename,
                                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    with tarfile.open(filepath, 'r:gz') as t:
        extracted_dir_path = os.path.join(
            dest_directory, t.getmembers()[0].name)
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(t, dest_directory)
    return extracted_dir_path


def main(argv=None):  # pylint: disable=unused-argument
    # bin_dataset_dir = cifar10.maybe_download_and_extract(cifar10.DATA_URL)
    python_dataset_dir = maybe_download_and_extract(
        DATA_URL_PYTHON)
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train_all, validate_all = cifar10_input.load_and_preprocess_input(
        dataset_dir=python_dataset_dir)
    train(train_all=train_all, validate_all=validate_all)


if __name__ == '__main__':
    tf.app.run()
