#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/16 16:01
# @Author : XU_Sheng
# @Site : 
# @File : character_7.py
# @Software: PyCharm


import tensorflow as tf
from tensorflow import keras
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
for i ,layer in enumerate(net.variables):
    print(i,layer.shape)


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
net.add(tf.keras.layers.Dense(20)) #20 是为了和后面相乘 与前一个模块对应
net.add(FancyMLP())
XXX = tf.random.uniform((3,5,8))
Y = net(XXX)
print('Y is: ',Y)

#打印当前模型参数 两种方法一致
for i ,layer in enumerate(net.variables):
    print(i,layer.shape)
for i ,layer in enumerate(net.weights):
    print(i,layer.shape)

class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(
            units=10,
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )
        self.d2 = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.ones_initializer(),
            bias_initializer=tf.ones_initializer()
        )

    def call(self, input):
        output = self.d1(input)
        output = self.d2(output)
        return output

X = tf.random.uniform((2,20))
net = Linear()
y =net(X)
weights = net.get_weights()
print('weights is: ',weights)
net.save_weights("1-7saved_model.h5")

net2 = Linear()
net2(X)
net2.load_weights("1-7saved_model.h5")
yy = net2(X)
print(y)
print(yy)