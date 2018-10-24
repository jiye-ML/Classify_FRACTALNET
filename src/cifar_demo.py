# -*- coding: utf-8 -*-

""" Convolutional network applied to CIFAR-10 dataset classification task.

References:
    Learning Multiple Layers of Features from Tiny Images, A. Krizhevsky, 2009.

Links:
    [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

"""
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, global_avg_pool
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.activations import softmax
from fractal_block import fractal_conv2d
from tensorflow.contrib import slim
from tflearn.layers.normalization import batch_normalization

# Data loading and preprocessing
from tflearn.datasets import cifar10
(X, Y), (X_test, Y_test) = cifar10.load_data()
X, Y = shuffle(X, Y)
Y = to_categorical(Y, 10)
Y_test = to_categorical(Y_test, 10)

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

# Convolutional network building
net = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

filters = [64,128,256,512]
for f in filters:
  net = fractal_conv2d(net, 4, f, 3,
                       normalizer_fn=batch_normalization)
  net = slim.max_pool2d(net,2, 2)

net = fractal_conv2d(net, 4, 512, 2,
                     normalizer_fn=batch_normalization)


net = conv_2d(net, 10, 1)
net = global_avg_pool(net)
net = softmax(net)

net = regression(net, optimizer='adam',
                     loss='categorical_crossentropy',
                 learning_rate=.002)

# Train using classifier
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=400, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=32, run_id='cifar10_cnn')
