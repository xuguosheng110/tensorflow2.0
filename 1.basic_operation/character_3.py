#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/13 18:55
# @Author : xuguosheng
# @contact: xgs11@qq.com
# @Site : 
# @File : character_3.py
# @Software: PyCharm
# encoding: utf-8

import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

num_inputs = 2
features = tf.random.normal((1000,2),mean=0,stddev=1)
true_w = [3,-2]
true_b = 1
label = true_w[0]*features[:,0]+true_w[1]*features[:,1] +true_b
label += tf.random.normal(label.shape,stddev=0.01)

print(features[0],label[0])

#迭代器数据读取 内存消耗小
def data_iter(batch_size,feature,labels):
    num_examples = len(feature)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        j = indices[i:min(i+batch_size,num_examples)]
        yield tf.gather(features,axis=0,indices=j), tf.gather(labels,axis=0,indices=j)

batch_size = 10
for X,y in data_iter(batch_size,features,label):
    break


def linreg(X,w,b):
    return tf.matmul(X,w)+b


def squared_losses(y_hat,y):
    return (y_hat-tf.reshape(y,y_hat.shape))**2/2


def sgd(params,lr,batch_size,grads):
    for i ,param in enumerate(params):
        param.assign_sub(lr*grads[i]/batch_size)

lr = 0.01
num_epoches = 3
net = linreg
loss = squared_losses

w = tf.Variable(tf.random.normal((2,1),stddev=0.01))
b = tf.Variable(tf.zeros((1,)))
for epoch in range(num_epoches):
    for X,y in data_iter(batch_size,features,label):
        with tf.GradientTape() as g:
            g.watch([w,b])
            l = loss(net(X,w,b),y)
        grads =  g.gradient(l,[w,b])
        sgd([w,b],lr,batch_size,grads)
    train_l = loss(net(features,w,b),label)
    print('epoch %d, loss %f' % (epoch +1,tf.reduce_mean(train_l)))

print(true_b,b)
print(true_w,w)