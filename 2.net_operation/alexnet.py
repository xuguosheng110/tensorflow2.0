#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/19 10:47
# @Author : XU_Sheng
# @Site : 
# @File : alexnet.py
# @Software: PyCharm
import tensorflow as tf
import numpy as np

Alexnet = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(3,3)   ,strides=2),
    tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=3,strides=2),
    tf.keras.layers.Conv2D(filters=384,kernel_size=3,padding='same',activation='relu'),
    tf.keras.layers.Conv2D(filters=384,kernel_size=3,padding='same',activation='relu'),
    tf.keras.layers.Conv2D(filters=256,kernel_size=3,padding='same',activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=3,strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4096,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4096,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10,activation='sigmoid')
])

x = tf.random.uniform((1,224,224,1))
y = Alexnet(x)
print(y)
# for layer in Alexnet.layers:
#     x = layer(x)
#     print(layer.name,x.shape)

class Data_set():
    def __init__(self):
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (self.train_img,self.train_label),(self.test_img,self.test_label)=fashion_mnist.load_data()
        self.train_img = np.expand_dims(self.train_img.astype(np.int32)/255.0,axis=-1)
        self.test_img = np.expand_dims(self.test_img.astype(np.int32),axis=-1)
        self.train_label = self.train_label.astype(np.int32)
        self.test_label = self.test_label.astype(np.int32)
        self.num_train,self.num_test = self.train_img.shape[0],self.test_img.shape[0]

    def get_batch_train(self,batch_size):
        index = np.random.randint(0,self.num_train,batch_size)
        resize_img = tf.image.resize_with_pad(self.train_img[index],224,224)
        return resize_img.numpy(),self.train_label[index]

    def get_batch_test(self,batch_size):
        index = np.random.randint(0,self.num_test,batch_size)
        resize_img = tf.image.resize_with_pad(self.test_img[index],224,224)
        return resize_img.numpy(),np.array(self.test_label[index])


optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
this_loss = tf.losses.SparseCategoricalCrossentropy()
Alexnet.compile(optimizer=optimizer,loss=this_loss,
                metrics=['accuracy'])
batch_size = 1024
data_set = Data_set()
# train_img,train_label = data_set.get_batch_train(batch_size)
# test_img,test_label = data_set.get_batch_test(batch_size)
# print(train_label.shape)
# print(train_img.shape)
# Alexnet.fit(train_img,train_label,epochs=10,batch_size=32,verbose=1)

def train_step():
    epoch = 10
    for i in range(epoch):
        x_batch, y_batch = data_set.get_batch_train(1024)
        Alexnet.fit(x_batch, y_batch,verbose=2)
        Alexnet.save_weights(str(i) + "xgs_alexnet_weights.h5")


# train_step()
x = tf.random.uniform((1,224,224,1))
y = Alexnet(x)
Alexnet.load_weights('xgs_alexnet_weights.h5')
x_batch, y_batch = data_set.get_batch_test(1024)
Alexnet.evaluate(x_batch,y_batch,verbose=2)