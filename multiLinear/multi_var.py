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

plt.plot(x_sample, y_sample, label='real curve')
plt.legend()
plt.show()







