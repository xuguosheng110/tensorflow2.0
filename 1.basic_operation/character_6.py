#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/16 12:00
# @Author : xgs11@qq.com
# @Site : 感知机
# @File : character_6.py
# @Software: PyCharm
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
batch_size = 256
x_train = tf.cast(x_train/255.,dtype=tf.float32)
x_test = tf.cast(x_test/255.,dtype=tf.float32)
train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
test_iter = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

num_inputs ,num_outputs,num_hiddens=784,10,256
W1 = tf.Variable(tf.random.normal(shape=(num_inputs,num_hiddens),mean=0,stddev=0.1,dtype=tf.float32))
b1 = tf.Variable(tf.zeros(shape=num_hiddens,dtype=tf.float32))
W2 = tf.Variable(tf.random.normal(shape=(num_hiddens,num_outputs),mean=0,stddev=0.01,dtype=tf.float32))
b2 = tf.Variable(tf.zeros(num_outputs,dtype=tf.float32))

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    # tf.keras.layers.Dropout(0.5),
    keras.layers.Dense(256,activation='relu'),
    # tf.keras.layers.Dropout(0.5),
    # keras.layers.Dense(64, activation='tanh'),
    # tf.keras.layers.Dropout(0.5),
    keras.layers.Dense(10,activation='softmax')])

optimizer = tf.keras.optimizers.SGD(lr=0.01)
model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=50,batch_size=1024,validation_data=(x_test,y_test),validation_freq=1)
model_ = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dropout(0.2),
    keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dropout(0.2),
    keras.layers.Dense(10,activation='softmax')])

optimizer = tf.keras.optimizers.SGD(lr=0.01)
model_.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model_.fit(x_train,y_train,epochs=50,batch_size=1024,validation_data=(x_test,y_test),validation_freq=1)

print(model.evaluate(x_test,y_test))
print(model_.evaluate(x_test,y_test))