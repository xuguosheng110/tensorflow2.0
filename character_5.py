# -*- coding: utf-8 -*-
#================================================================
# Copyright (C) 2020 * Ltd. All rights reserved.
# Time : 2020/3/15 18:01
# Author : Xuguosheng
# contact: xgs11@qq.com
# File : character_5.py
# Software: PyCharm
# Description :图像分类 fashsion_minist
#================================================================

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
import cv2
import os

# os.environ['CUDA_DEVICE_ORDER'] = '-1'

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def softmax(logits, axis=-1):
    return tf.exp(logits)/tf.reduce_sum(tf.exp(logits), axis, keepdims=True)


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0  # 0表示不用额外的进程来加速读取数据
else:
    num_workers = 4

x_train = tf.cast(x_train/255.,dtype=tf.float32)
x_test = tf.cast(x_test/255.,dtype=tf.float32)
train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
test_iter = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

# num_inputs = 784
# num_outputs = 10
# W = tf.Variable(tf.random.normal(shape=(num_inputs,num_outputs),dtype=tf.float32))
# b = tf.Variable(tf.zeros(num_outputs,dtype=tf.float32))
num_inputs = 784
num_outputs = 10
W = tf.Variable(tf.random.normal(shape=(num_inputs, num_outputs), mean=0, stddev=0.01, dtype=tf.float32))
b = tf.Variable(tf.zeros(num_outputs, dtype=tf.float32))

def net(X):
    xx = tf.reshape(X, shape=(-1, W.shape[0]))
    logits = tf.matmul(xx, W)
    logits +=b
    return softmax(logits)


def cross_entropy(y_hat, y):
    y = tf.cast(tf.reshape(y, shape=[-1, 1]),dtype=tf.int32)
    y = tf.one_hot(y, depth=y_hat.shape[-1])
    y = tf.cast(tf.reshape(y, shape=[-1, y_hat.shape[-1]]),dtype=tf.int32)
    return -tf.math.log(tf.boolean_mask(y_hat, y)+1e-8)

# y_hat = np.array([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# y = np.array([0, 2], dtype='int32')
# z = tf.boolean_mask(y_hat, tf.one_hot(y, depth=3))

def accuracy(y_hat, y):
    return np.mean((tf.argmax(y_hat, axis=1) == y))#判定最大值索引与结果比较

def evaluate_accuracy(data_iter,net):
    acc_sum,n =0.0 ,0
    for x,y in data_iter:
        y = tf.cast(y,dtype=tf.int32)
        acc_sum +=np.sum(tf.cast(tf.argmax(net(x),axis =1),dtype=tf.int32)==y)
        n +=y.shape[0]
    print(n)
    return acc_sum/n

num_epochs, lr = 5, 0.1
# 本函数已保存在d2lzh包中方便以后使用
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, trainer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                y_hat = net(X)
                l = tf.reduce_sum(loss(y_hat, y))
            grads = tape.gradient(l, params)
            if trainer is None:
                # 如果没有传入优化器，则使用原先编写的小批量随机梯度下降
                for i, param in enumerate(params):
                    param.assign_sub(lr * grads[i] / batch_size)
            else:
                # tf.keras.optimizers.SGD 直接使用是随机梯度下降 theta(t+1) = theta(t) - learning_rate * gradient
                # 这里使用批量梯度下降，需要对梯度除以 batch_size, 对应原书代码的 trainer.step(batch_size)
                trainer.apply_gradients(zip([grad / batch_size for grad in grads], params))
                # trainer.apply_gradients(zip(grads/batch_size, params))


            y = tf.cast(y, dtype=tf.float32)
            train_l_sum += l.numpy()
            train_acc_sum += tf.reduce_sum(tf.cast(tf.argmax(y_hat, axis=1) == tf.cast(y, dtype=tf.int64), dtype=tf.int64)).numpy()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'% (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
start = time.time()
trainer = tf.keras.optimizers.SGD(lr)
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr,trainer)
print('use time is: ',time.time()-start)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
# 数据读取与测试，28 *28的灰度图
# feature,label=x_train[0],y_train[0]
# print(type(x_test),type(y_test))
# print(feature,label)
# cv2.imshow('first_img',feature)
# cv2.waitKey(0)