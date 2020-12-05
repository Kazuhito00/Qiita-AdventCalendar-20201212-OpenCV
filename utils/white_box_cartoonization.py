#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import tensorflow.compat.v1 as tf
import tf_slim as slim


def resblock(inputs, out_channel=32, name='resblock'):

    with tf.variable_scope(name):

        x = slim.convolution2d(inputs,
                               out_channel, [3, 3],
                               activation_fn=None,
                               scope='conv1')
        x = tf.nn.leaky_relu(x)
        x = slim.convolution2d(x,
                               out_channel, [3, 3],
                               activation_fn=None,
                               scope='conv2')

        return x + inputs


def unet_generator(inputs,
                   channel=32,
                   num_blocks=4,
                   name='generator',
                   reuse=False):
    with tf.variable_scope(name, reuse=reuse):

        x0 = slim.convolution2d(inputs, channel, [7, 7], activation_fn=None)
        x0 = tf.nn.leaky_relu(x0)

        x1 = slim.convolution2d(x0,
                                channel, [3, 3],
                                stride=2,
                                activation_fn=None)
        x1 = tf.nn.leaky_relu(x1)
        x1 = slim.convolution2d(x1, channel * 2, [3, 3], activation_fn=None)
        x1 = tf.nn.leaky_relu(x1)

        x2 = slim.convolution2d(x1,
                                channel * 2, [3, 3],
                                stride=2,
                                activation_fn=None)
        x2 = tf.nn.leaky_relu(x2)
        x2 = slim.convolution2d(x2, channel * 4, [3, 3], activation_fn=None)
        x2 = tf.nn.leaky_relu(x2)

        for idx in range(num_blocks):
            x2 = resblock(x2,
                          out_channel=channel * 4,
                          name='block_{}'.format(idx))

        x2 = slim.convolution2d(x2, channel * 2, [3, 3], activation_fn=None)
        x2 = tf.nn.leaky_relu(x2)

        h1, w1 = tf.shape(x2)[1], tf.shape(x2)[2]
        x3 = tf.image.resize_bilinear(x2, (h1 * 2, w1 * 2))
        x3 = slim.convolution2d(x3 + x1,
                                channel * 2, [3, 3],
                                activation_fn=None)
        x3 = tf.nn.leaky_relu(x3)
        x3 = slim.convolution2d(x3, channel, [3, 3], activation_fn=None)
        x3 = tf.nn.leaky_relu(x3)

        h2, w2 = tf.shape(x3)[1], tf.shape(x3)[2]
        x4 = tf.image.resize_bilinear(x3, (h2 * 2, w2 * 2))
        x4 = slim.convolution2d(x4 + x0, channel, [3, 3], activation_fn=None)
        x4 = tf.nn.leaky_relu(x4)
        x4 = slim.convolution2d(x4, 3, [7, 7], activation_fn=None)

        return x4


def tf_box_filter(x, r):
    k_size = int(2 * r + 1)
    ch = x.get_shape().as_list()[-1]
    weight = 1 / (k_size**2)
    box_kernel = weight * np.ones((k_size, k_size, ch, 1))
    box_kernel = np.array(box_kernel).astype(np.float32)
    output = tf.nn.depthwise_conv2d(x, box_kernel, [1, 1, 1, 1], 'SAME')
    return output


def guided_filter(x, y, r, eps=1e-2):

    x_shape = tf.shape(x)

    N = tf_box_filter(tf.ones((1, x_shape[1], x_shape[2], 1), dtype=x.dtype),
                      r)

    mean_x = tf_box_filter(x, r) / N
    mean_y = tf_box_filter(y, r) / N
    cov_xy = tf_box_filter(x * y, r) / N - mean_x * mean_y
    var_x = tf_box_filter(x * x, r) / N - mean_x * mean_x

    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    mean_A = tf_box_filter(A, r) / N
    mean_b = tf_box_filter(b, r) / N

    output = mean_A * x + mean_b

    return output


def fast_guided_filter(lr_x, lr_y, hr_x, r=1, eps=1e-8):
    lr_x_shape = tf.shape(lr_x)
    hr_x_shape = tf.shape(hr_x)

    N = tf_box_filter(
        tf.ones((1, lr_x_shape[1], lr_x_shape[2], 1), dtype=lr_x.dtype), r)

    mean_x = tf_box_filter(lr_x, r) / N
    mean_y = tf_box_filter(lr_y, r) / N
    cov_xy = tf_box_filter(lr_x * lr_y, r) / N - mean_x * mean_y
    var_x = tf_box_filter(lr_x * lr_x, r) / N - mean_x * mean_x

    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    mean_A = tf.image.resize_images(A, hr_x_shape[1:3])
    mean_b = tf.image.resize_images(b, hr_x_shape[1:3])

    output = mean_A * hr_x + mean_b

    return output


if __name__ == '__main__':

    pass