#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/16 16:01
# @Author : XU_Sheng
# @Site : 
# @File : character_7.py
# @Software: PyCharm
# import tensorflow as tf
# import numpy as np
# from tensorflow import keras, nn, losses
# from tensorflow.keras.layers import Dropout, Flatten, Dense

import tensorflow as tf
import numpy as np
#简单模型
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dens1   = tf.keras.layers.Dense(units=256,activation='relu')
        self.dens2   = tf.keras.layers.Dense(units=10)

    def __call__(self, inputs):
        x = self.flatten(inputs)
        x = self.dens1(x)
        x = self.dens2(x)
        return x

X = tf.random.uniform((2,28))
model = MLP()
y =model(X)
print('y is: ',y)
#带常量模型
class FancyMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.rand_weight = tf.constant(tf.random.uniform((20,20)))
        self.dense = tf.keras.layers.Dense(units=20,activation='relu')

    def call(self,inputs):
        x = self.flatten(inputs)
        x = tf.nn.relu(tf.matmul(x,self.rand_weight)+1)
        x = self.dense(x)
        x = self.dense(x)
        while tf.norm(x)>1:  #向量 张量 矩阵等的范数
            x /= 2
        if tf.norm(x) < 0.8:
            x *= 10
        return tf.reduce_sum(x)

XX = tf.random.uniform((2,4,5))
net = FancyMLP()
z = net(XX)
print('z is: ',z)
print('xxx is: ',net.variables)


class Fancy_MLP(tf.keras.Model):
    def __init__(self):
        super(Fancy_MLP, self).__init__()
        self.net = tf.keras.Sequential()
        self.net.add(tf.keras.layers.Flatten())
        self.net.add(tf.keras.layers.Dense(64,activation='relu'))
        self.net.add(tf.keras.layers.Dense(32,activation='relu'))
        self.Dense = tf.keras.layers.Dense(16,activation='relu')

    def call(self, inputs):
        return self.Dense(self.net(inputs))

#模型套模型
net = tf.keras.Sequential()
net.add(Fancy_MLP())
net.add(tf.keras.layers.Dense(20))
net.add(FancyMLP())
Y = net(XX)
print('Y is: ',Y)
