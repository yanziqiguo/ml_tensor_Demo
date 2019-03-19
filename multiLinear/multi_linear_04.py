# -*- coding: utf-8 -*-
# !/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#定义参数
w_target = np.array([0.5, 3, 2.4])
b_target = np.array([0.9])

f_des = 'y = {:.2f} + {:.2f} * x + {:.2f} * x^2 +{:.2f} * x^3'.format(
    b_target[0], w_target[0], w_target[1], w_target[2]
)
print(f_des)

x_sample = np.arange(-3,3.1,0.1)
y_sample = b_target[0] + w_target[0]*x_sample + w_target[1]*x_sample**2 + w_target[2]*x_sample**3

x_train = np.stack([x_sample**i for i in range(1,4)], axis=1)
x_train = tf.constant(x_train, dtype=tf.float32, name='x_train')
y_train = tf.constant(y_sample, dtype=tf.float32, name='y_train')

w = tf.Variable(initial_value=tf.random_normal(shape=(3,1)), dtype=tf.float32, name='weights')
b = tf.Variable(initial_value=0, dtype=tf.float32, name='bias')

def multi_linear(x):
    return tf.squeeze(tf.matmul(x,w) + b)

y_ = multi_linear(x_train)

sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())

loss = tf.reduce_mean(tf.square(y_train-y_))
loss_numpy = sess.run(loss)
print(loss_numpy)

w_grad,b_grad = tf.gradients(loss,[w,b])

print(w_grad.eval(session=sess))
print(b_grad.eval(session=sess))

lr = 1e-3

w_update = w.assign_sub(lr*w_grad)
b_update = b.assign_sub(lr*b_grad)

sess.run([w_update,b_update])
sess.run(tf.global_variables_initializer())

x_train_value = x_train.eval(session=sess)
y_train_value = y_train.eval(session=sess)
y_pred_value = y_.eval(session=sess)
loss_numpy=loss.eval(session=sess)

for e in range(100):
    sess.run([w_update,b_update])
    x_train_value = x_train.eval(session=sess)
    y_train_value = y_train.eval(session=sess)
    y_pred_value = y_.eval(session=sess)
    loss_numpy=loss.eval(session=sess)

    if(e+1)%20==0:
        print('epoch: {}, loss: {}'.format(e+1, loss_numpy))

plt.plot(x_train_value[:,0], y_pred_value, label='fitting curve', color='r')
plt.plot(x_train_value[:,0], y_train_value, label='real curve', color='b')
plt.legend()
plt.title('loss: %.4f' % loss_numpy)
plt.show()

sess.close()





