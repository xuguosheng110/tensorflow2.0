#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/18 15:23
# @Author : XU_Sheng
# @Site : 
# @File : 1-lenet.py
# @Software: PyCharm
import tensorflow as tf
import numpy as np

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = tf.reshape(train_images, (train_images.shape[0],train_images.shape[1],train_images.shape[2], 1))
test_images = tf.reshape(test_images, (test_images.shape[0],test_images.shape[1],test_images.shape[2], 1))
lenet = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=6,kernel_size=5,activation='sigmoid',input_shape=(28,28,1)),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=16,kernel_size=5,activation='sigmoid'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120,activation='sigmoid'),
    tf.keras.layers.Dense(84,activation='sigmoid'),
    tf.keras.layers.Dense(10,activation='sigmoid') #最终有十种类别
])

#错误 不能一次add多个网络层，只能一个个添加
# lenet.add([
#     tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid', input_shape=(28, 28, 1)),
#     tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
#     # tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation='sigmoid'),
#     # tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
#     # tf.keras.layers.Flatten(),
#     # tf.keras.layers.Dense(120, activation='sigmoid'),
#     # tf.keras.layers.Dense(84, activation='sigmoid'),
#     # tf.keras.layers.Dense(10, activation='sigmoid')
# ])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.9, momentum=0.0, nesterov=False)
lenet.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
lenet.fit(train_images, train_labels, epochs=3, validation_split=0.1,batch_size=1000,verbose=2)#不指定 batchsize就是32
acc = lenet.evaluate(test_images,test_labels,verbose=1)
lenet.save('xgs_lenet')
print(test_labels.shape)
print(acc)
model = tf.keras.models.load_model('xgs_lenet')
# x = tf.random.uniform((15,28,28,1))
# y = tf.range(15)
# y = tf.cast(y,dtype=tf.float32)
# z = lenet(x)
# acc = lenet.evaluate(x,[y])
# result = lenet.predict(x)
# print('result is:',result)
