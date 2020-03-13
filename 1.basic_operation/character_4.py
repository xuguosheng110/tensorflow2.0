#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/13 22:08
# @Author : xuguosheng
# @contact: xgs11@qq.com
# @Site :  线性模型建立
# @File : character_4.py
# @Software: PyCharm
# encoding: utf-8
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras

features = tf.random.normal((1000,2),mean=0,stddev=1)
true_w = [3,-2]
true_b = 1
labels = true_w[0]*features[:,0]+true_w[1]*features[:,1] +true_b
labels += tf.random.normal(labels.shape,stddev=0.01)

dataset = tf.data.Dataset.from_tensor_slices((features,labels))
dataset = dataset.shuffle(buffer_size = 1000) #buffer 数目大于等于样本数 随机打乱
dataset = dataset.batch(batch_size=10)
# data_iter = iter(dataset)

model = keras.Sequential()
model.add(keras.layers.Dense(1,kernel_initializer = tf.initializers.RandomNormal(stddev=0.01)))
loss = tf.losses.MeanSquaredError()
optimizer = keras.optimizers.SGD(learning_rate=0.03)
# @tf.function
for epoch in range(10):
    l_list =[]
    for (batch ,(x,y)) in enumerate(dataset):
        with tf.GradientTape() as g:
            l = loss(model(x,training=True), y)
        gradient = g.gradient(l, model.trainable_variables)
        optimizer.apply_gradients(zip(gradient,model.trainable_variables))
        l_list.append(l)
    l_list = np.array(l_list)
    print("epoch is %d ,loss is %f"%(epoch,l_list.mean()))

