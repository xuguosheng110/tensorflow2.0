# -*- coding: utf-8 -*-
#================================================================
# Copyright (C) 2020 * Ltd. All rights reserved.
# Time : 2020/3/18 0:20
# Author : Xuguosheng
# contact: xgs11@qq.com
# File : character_8.py
# Software: PyCharm
# Description :
#================================================================
import tensorflow as tf


class myDense(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):     # 这里 input_shape 是第一次运行call()时参数inputs的形状
        self.w = self.add_weight(name='w',
            shape=[input_shape[-1], self.units], initializer=tf.random_normal_initializer())
        self.b = self.add_weight(name='b',
            shape=[self.units], initializer=tf.zeros_initializer())

    def call(self, inputs):
        y_pred = tf.matmul(inputs, self.w) + self.b
        return y_pred


X = tf.random.uniform((2,20))
dense = myDense(3)
y = dense(X)
weights = dense.get_weights()
print(y)
for weight in weights:
    print(weight.shape)

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print("可用的GPU：",gpus,"\n可用的CPU：", cpus)