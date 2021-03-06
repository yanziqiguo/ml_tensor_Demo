# -*- coding: utf-8 -*-
# !/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# read data from file
with open('./logistic_regression.txt', 'r') as f:
    data_list = [i.split('\n')[0].split(',') for i in f.readlines()]
    data = [(float(i[0]), float(i[1]), float(i[2])) for i in data_list]

# 标准化
x0_max = max([i[0] for i in data])
x1_max = max([i[1] for i in data])
data = [(i[0] / x0_max, i[1] / x1_max, i[2]) for i in data]

x0 = list(filter(lambda x: x[-1] == 0.0, data))
x1 = list(filter(lambda x: x[-1] == 1.0, data))

plot_x0 = [i[0] for i in x0]
plot_y0 = [i[1] for i in x0]
plot_x1 = [i[0] for i in x1]
plot_y1 = [i[1] for i in x1]

np_data = np.array(data, dtype='float32')
x_data = tf.constant(np_data[:, 0:2], name='x')
y_data = tf.expand_dims(tf.constant(np_data[:, -1]), axis=-1)

w = tf.get_variable(initializer=tf.random_normal_initializer(seed=2017), shape=(2, 1), name='weights')
b = tf.get_variable(initializer=tf.zeros_initializer(), shape=(1), name='bias')


def logistic_regression(x):
    return tf.sigmoid(tf.matmul(x, w) + b)


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


def binary_loss(y_pred, y):
    logit = tf.reduce_mean(y * tf.log(y_pred) + (1 - y) * tf.log(1 - y_pred))
    return -logit


y_pred = logistic_regression(x_data)
loss = binary_loss(y_pred, y_data)

print(loss.eval(session=sess))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1, name='optimizer')

train_op = optimizer.minimize(loss)

import time

start = time.time()
for e in range(1000):
    sess.run(train_op)

    if (e + 1) % 200 == 0:
        y_true_label = y_data.eval(session=sess)
        y_pred_numpy = y_pred.eval(session=sess)
        y_pred_label = np.greater_equal(y_pred_numpy, 0.5).astype(np.float32)
        accuracy = np.mean(y_pred_label == y_true_label)
        loss_numpy = loss.eval(session=sess)
        print('Epoch %d,Loss: %.4f, Acc: %.4f' % (e+1, loss_numpy, accuracy))

print()
print('manual_GD cost time: %4.f' % (time.time() - start))

w_numpy = w.eval(session=sess)
b_numpy = b.eval(session=sess)

w0 = w_numpy[0]
w1 = w_numpy[1]
b0 = b_numpy[0]

plot_x = np.arange(0.2, 1, 0.01)
plot_y = (-w0 * plot_x - b0) / w1

plt.plot(plot_x, plot_y, 'g', label='cutting line')
plt.plot(plot_x0, plot_y0, 'ro', label='x_0')
plt.plot(plot_x1, plot_y1, 'bo', label='x_1')

plt.legend(loc='best')
plt.show()
