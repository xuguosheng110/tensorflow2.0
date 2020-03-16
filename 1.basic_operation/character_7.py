#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/3/16 16:01
# @Author : XU_Sheng
# @Site : 
# @File : character_7.py
# @Software: PyCharm
import tensorflow as tf
import numpy as np
from tensorflow import keras, nn, losses
from tensorflow.keras.layers import Dropout, Flatten, Dense


def dropout(X, drop_prob):
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    # 这种情况下把全部元素都丢弃
    if keep_prob == 0:
        return tf.zeros_like(X)
    #初始mask为一个bool型数组，故需要强制类型转换
    mask = tf.random.uniform(shape=X.shape, minval=0, maxval=1) < keep_prob
    print(mask)
    return tf.cast(mask, dtype=tf.float32) * tf.cast(X, dtype=tf.float32) / keep_prob

X = tf.reshape(tf.range(0,20),shape=(5,4))
print(X)
print(dropout(X,1))
print(dropout(X,0.2))
print(dropout(X,0.5))